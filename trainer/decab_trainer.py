
import logging
from operator import mod
from tkinter import W
import torch 
from utils import Meters
import torch.nn as nn
import argparse
import copy
import torch.backends.cudnn as cudnn   
from config.defaults import update_config,_C as cfg
import numpy as np 
from utils.build_optimizer import get_optimizer
import models 
import time 
import torch.optim as optim 
import os   
import datetime
import torch.nn.functional as F  
from .base_trainer import BaseTrainer
import models 
from utils import FusionMatrix
from models.projector import  Projector 
from dataset.base import BaseNumpyDataset
from dataset.build_dataloader import _build_loader
from utils.misc import AverageMeter  
from loss.debiased_soft_contra_loss import * 
from models.feature_queue import FeatureQueue 

class DeCABTrainer(BaseTrainer):   
    def __init__(self, cfg):        
        super().__init__(cfg)     
        
        self.lambda_d=cfg.ALGORITHM.DeCAB.LAMBDA_D 
        self.final_lambda_d=self.lambda_d 
        self.m=cfg.ALGORITHM.DeCAB.M 
        self.debiased_contra_temperture=cfg.ALGORITHM.DeCAB.DeCAB_CONTRA_TEMPERTURE        
        self.ood_detect_confusion_matrix=OODDetectFusionMatrix(self.num_classes)
        self.loss_contrast= DebiasSoftConLoss(temperature=self.debiased_contra_temperture)
     
        self.warmup_epoch=self.cfg.ALGORITHM.DeCAB.WARMUP_EPOCH        
               
        self.data_dist=self.labeled_trainloader.dataset.num_per_cls_list
        self.cls_prob=torch.tensor(self.data_dist/self.data_dist[0]).cuda() 
        self.class_thresh=0.5+self.cls_prob*(self.conf_thres-0.5)
        self.contrast_with_thresh=cfg.ALGORITHM.DeCAB.CONTRAST_THRESH 
        
        if cfg.RESUME!='':
            self.load_checkpoint(cfg.RESUME) 
        
    
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter() 
        self.losses_d_ctr=AverageMeter()
        
   
    def train_step(self,pretraining=False):  
        self.model.train()
        loss =0 
        try:        
            data_x = self.labeled_train_iter.next() 
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader)
            data_x = self.labeled_train_iter.next() 
        try:       
            data_u = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter=iter(self.unlabeled_trainloader)
            data_u = self.unlabeled_train_iter.next() 
            
        inputs_x_w=data_x[0] 
        targets_x=data_x[1]
         
        inputs_u_w=data_u[0][0]
        inputs_u_s=data_u[0][1]
        inputs_u_s1=data_u[0][2]
        u_index=data_u[2]
        u_index=u_index.long().cuda()
        
        if isinstance(inputs_x_w,list):
            inputs_x_w=inputs_x_w[0]
        
        inputs = torch.cat(
                [inputs_x_w,inputs_u_w, inputs_u_s, inputs_u_s1],
                dim=0).cuda()
        
        targets_x=targets_x.long().cuda()
        
        encoding = self.model(inputs,return_encoding=True)
        features=self.model(encoding,return_projected_feature=True)
        logits=self.model(encoding,classifier=True)
        batch_size=inputs_x_w.size(0)
        logits_x = logits[:batch_size]
         
        logits_u_w, logits_u_s, _ = logits[batch_size:].chunk(3)
        f_l_w= features[:batch_size]
        f_u_w, f_u_s1, f_u_s2 = features[batch_size:].chunk(3)
        
        # 1. ce loss           
        loss_cls=self.l_criterion(logits_x, targets_x)
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)  
         
        # 2. cons loss 
        # filter out samples by class-aware threshold 
        with torch.no_grad(): 
            probs_u_w = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, pred_class = torch.max(probs_u_w, dim=-1)          
        
        if self.class_aware_thresh_enable:
            loss_weight = max_probs.ge(self.class_thresh[pred_class]).float() 
        else:
            loss_weight = max_probs.ge(self.conf_thres).float()     
        
        loss_cons = self.ul_criterion(
            logits_u_s, pred_class, weight=loss_weight, avg_factor=logits_u_s.size(0)
        )
        labels = pred_class 
        
        # 3. ctr loss  
        contrast_mask = max_probs.ge(self.contrast_with_thresh).float()
        features = torch.cat([f_u_s1.unsqueeze(1), f_u_s2.unsqueeze(1)], dim=1)  
            
        if self.sample_weight_enable:
            sample_weight=conf_sample*(1-max_probs)                        
        else:
            sample_weight=conf_sample 
        if self.epoch>self.warmup_epoch: 
            with torch.no_grad():  
                f=encoding[batch_size:batch_size+inputs_u_w.size(0)].detach()
                cos_sim= cosine_similarity(f.detach().cpu().numpy()) 
                cos_sim = torch.from_numpy(cos_sim).cuda()                    
                y = labels.contiguous().view(-1, 1)
                labeled_mask= torch.eq(y, y.T).float() 
                mask=(1-cos_sim)*labeled_mask + cos_sim*(1-labeled_mask) 
            loss_d_ctr = self.loss_contrast(features,  
                                            mask=mask,                                                  
                                            reduction=None)
            loss_d_ctr = (loss_d_ctr * sample_weight).mean()  
        else:loss_d_ctr=torch.tensor(0.).cuda()  
             
        loss=loss_cls+loss_cons+self.lambda_d*loss_d_ctr
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  
        self.losses_d_ctr.update(loss_d_ctr.item(),labels.shape[0]) 
        self.losses_x.update(loss_cls.item(), batch_size)
        self.losses_u.update(loss_cons.item(), inputs_u_s.size(0)) 
        self.losses.update(loss.item(),batch_size)
        
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} Avg_Loss_c:{:>5.4f} =='\
                .format(self.epoch,self.iter%self.step_per_epoch if self.iter%self.step_per_epoch>0 else self.step_per_epoch,
                        self.step_per_epoch,
                        self.losses.avg,self.losses_x.avg,self.losses_u.avg,self.losses_d_ctr.avg))
             
        return now_result.cpu().numpy(), targets_x.cpu().numpy()
    