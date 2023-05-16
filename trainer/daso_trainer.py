
import logging
from operator import mod
from tkinter import W
import torch 
from utils import Meters
import torch.nn as nn
import argparse
import torch.backends.cudnn as cudnn   
from config.defaults import update_config,_C as cfg
import numpy as np 
import models 
import time 
import torch.optim as optim
from models.feature_queue import FeatureQueue
import os   
import datetime
import torch.nn.functional as F  
from .base_trainer import BaseTrainer
from utils import DistributionLogger,FusionMatrix

class DASOTrainer(BaseTrainer):
    def __init__(self, cfg):
        
        super().__init__(cfg)      
    
        self.psa_loss_weight=cfg.ALGORITHM.DASO.PSA_LOSS_WEIGHT 
        self.queue = FeatureQueue(cfg, classwise_max_size=None, bal_queue=True)      
        
        self.similarity_fn = nn.CosineSimilarity(dim=2)
        self.is_ul_unknown = cfg.DATASET.NAME == "stl10"
        self.dist_logger = DistributionLogger(
                Meters(), num_classes=self.num_classes, is_ul_unknown=self.is_ul_unknown
            )        
        self.dist_logger.accumulate_pl = True
        class_count = self.get_label_dist()
        self.target_dist = class_count / class_count.sum()  # probability
        self.bal_param = class_count[-1] / class_count  # bernoulli parameter
        self.T_proto = cfg.ALGORITHM.DASO.PROTO_TEMP
        self.T_dist = cfg.ALGORITHM.DASO.DIST_TEMP        
        self.with_dist_aware = cfg.ALGORITHM.DASO.WITH_DIST_AWARE
        self.interp_alpha = cfg.ALGORITHM.DASO.INTERP_ALPHA        
        self.daso_warmup_iter=cfg.ALGORITHM.DASO.WARMUP_ITER      
    
            
    def train_step(self,pretraining=False):
        self.model.train()
        loss =0
        try:
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader)
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
 
        try:                 
            data = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter=iter(self.unlabeled_trainloader)
            data = self.unlabeled_train_iter.next()
        inputs_u=data[0][0]
        inputs_u2=data[0][1]
        ul_y=data[1] 
        
        inputs_x, targets_x = inputs_x.cuda(), targets_x.long().cuda(non_blocking=True)
        
        inputs_u , inputs_u2= inputs_u.cuda(),inputs_u2.cuda()
        
        logger_dict = {"gt_labels": targets_x, "ul_labels": torch.full_like(ul_y, -1).cuda() }  # initial log
        # push memory queue
        num_labels = targets_x.size(0)
        with torch.no_grad():
            l_feats = self.model(inputs_x, return_encoding=True) 
            self.queue.enqueue(l_feats.clone().detach(), targets_x.clone().detach())
            
            
        # feature vectors
        x=torch.cat((inputs_x,inputs_u,inputs_u2),dim=0)
        x = self.model(x,return_encoding=True)

        # initial empty assignment
        assignment = torch.Tensor([-1 for _ in range(inputs_u.size(0))]).float().cuda()
        if self.daso_warmup_iter<self.iter: 
            prototypes = self.queue.prototypes  # (K, D)
            feats_weak, feats_strong = x[num_labels:].chunk(2)  # (B, D)

            with torch.no_grad():
                # similarity between weak features and prototypes  (B, K)
                sim_weak = self.similarity_fn(
                    feats_weak.unsqueeze(1), prototypes.unsqueeze(0)
                ) / self.T_proto
                soft_target = sim_weak.softmax(dim=1)
                assign_confidence, assignment = torch.max(soft_target.detach(), dim=1)

            # soft loss
            if self.psa_loss_weight > 0:
                # similarity between strong features and prototypes  (B, K)
                sim_strong = self.similarity_fn(
                    feats_strong.unsqueeze(1), prototypes.unsqueeze(0)
                ) / self.T_proto

                loss_assign = self.psa_loss_weight * -1 * torch.sum(soft_target * F.log_softmax(sim_strong, dim=1),
                                            dim=1).sum() / sim_weak.size(0) 
                loss+=loss_assign
        logger_dict.update({"sem_pl": assignment})  # semantic pl
        # fixmatch pipelines
        logits_concat = self.model(x,classifier=True)
        logits_x = logits_concat[:num_labels]

        # loss computation 
        lx=self.l_criterion(logits_x, targets_x.long()) 
        # compute 1st branch accuracy
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)         
        logits_weak, logits_strong = logits_concat[num_labels:].chunk(2)
        with torch.no_grad():
            # compute pseudo-label
            p = logits_weak.softmax(dim=1)  # soft pseudo labels
            confidence, pred_class = torch.max(p.detach(), dim=1)  # (B, 1) 

            logger_dict.update({"linear_pl": pred_class})  # linear pl
            if self.daso_warmup_iter<self.iter: 
                current_pl_dist = self.dist_logger.get_pl_dist().cuda()  # (1, C)
                current_pl_dist = current_pl_dist**(1. / self.T_dist)
                current_pl_dist = current_pl_dist / current_pl_dist.sum()
                current_pl_dist = current_pl_dist / current_pl_dist.max()  # MIXUP

                pred_to_dist = current_pl_dist[pred_class].view(-1, 1)  # (B, )
                if not self.with_dist_aware:
                    pred_to_dist = self.interp_alpha  # override to fixed constant

                # pl mixup
                p = (1. - pred_to_dist) * p + pred_to_dist * soft_target

            confidence, pred_class = torch.max(p.detach(), dim=1)  # final pl             
            logger_dict.update({"pseudo_labels": pred_class, "pl_confidence": confidence})
            self.dist_logger.accumulate(logger_dict)
            self.dist_logger.push_pl_list(pred_class)

            loss_weight = confidence.ge(self.conf_thres).float()

        lu = self.ul_criterion(
            logits_strong, pred_class, weight=loss_weight, avg_factor=pred_class.size(0)
        ) 
        loss+=lx+lu
        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(lx.item(), inputs_x.size(0))
        self.losses_u.update(lu.item(), inputs_u.size(0)) 

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='.format(self.epoch,self.iter%self.step_per_epoch if self.iter%self.step_per_epoch>0 else self.step_per_epoch,self.step_per_epoch,self.losses.val,self.losses_x.avg,self.losses_u.val))
        
        return now_result.cpu().numpy(), targets_x.cpu().numpy()
 
    