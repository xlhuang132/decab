
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
import os   
import datetime
import torch.nn.functional as F  
from .fixmatch_trainer import FixMatchTrainer
from dataset.base import BaseNumpyDataset
from utils import FusionMatrix
from dataset.build_dataloader import _build_loader

class CReSTTrainer(FixMatchTrainer):    
    def __init__(self, cfg):
        super().__init__(cfg)         
        self.gen_period_steps=cfg.ALGORITHM.CREST.GEN_PERIOD_EPOCH * self.step_per_epoch
        self.t_min = cfg.ALGORITHM.CREST.TMIN
        self.with_progressive = cfg.ALGORITHM.CREST.PROGRESSIVE_ALIGN

        # unlabeled dataset configuration
        ul_dataset = self.unlabeled_trainloader.dataset
        ul_test_dataset = BaseNumpyDataset(
            ul_dataset.select_dataset(),
            transforms=self.test_loader.dataset.transforms, 
            num_classes=self.num_classes
            # is_ul_unknown=ul_dataset.is_ul_unknown
        )
        self.ul_test_loader = _build_loader(
            self.cfg, ul_test_dataset, is_train=False, has_label=False
        )

        # save init stats
        l_dataset = self.labeled_trainloader.dataset
        self.init_l_data=l_dataset
        # self.l_transforms=self.labeled_trainloader.dataset.transform.transforms
        self.init_l_data, self.l_transforms = l_dataset.select_dataset(return_transforms=True)
        self.current_l_dataset = l_dataset

        crest_alpha = cfg.ALGORITHM.CREST.ALPHA
        self.mu_per_cls = torch.pow(
            self.current_label_dist( normalize="max").clone(), (1 / crest_alpha)
        )

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
        
        inputs_x, targets_x = inputs_x.cuda(), targets_x.long().cuda(non_blocking=True)        
        inputs_u , inputs_u2= inputs_u.cuda(),inputs_u2.cuda()          
        x=torch.cat((inputs_x,inputs_u,inputs_u2),dim=0) 
        
        # fixmatch pipelines
        logits_concat = self.model(x)
        num_labels=inputs_x.size(0)
        logits_x = logits_concat[:num_labels]

        # loss computation 
        lx=self.l_criterion(logits_x, targets_x) 
        # compute 1st branch accuracy
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)         
        logits_weak, logits_strong = logits_concat[num_labels:].chunk(2)
        with torch.no_grad():
            # compute pseudo-label
            p = logits_weak.softmax(dim=1)  # soft pseudo labels
            confidence, pred_class = torch.max(p.detach(), dim=1) 
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

    def current_label_dist(self, **kwargs):
        return self.get_label_dist(dataset=self.current_l_dataset, **kwargs)
    
    def operate_after_epoch(self):
        if self.iter>self.warmup_iter and self.iter % self.gen_period_steps==0:
            self._build_new_generation()
        self.logger.info("=="*30)
        
        
    def _build_new_generation(self):
        print()
        self.logger.info(
            "{} iters -> {}-th generation".format(
                self.iter + 1, (self.iter + 1) // self.gen_period_steps + 1
            )
        )
        self._rebuild_labeled_dataset()
        self._rebuild_models()
        self._rebuild_optimizer(self.model) 
        
    def eval_ul_dataset(self):
        self.logger.info("evaluating ul data as test set...")
        ul_dataset = self.ul_test_loader.dataset
        ul_preds = torch.zeros(len(ul_dataset), self.num_classes)

        model = self.get_val_model()
        model.eval()
        with torch.no_grad():
            for i, (images, _, inds) in enumerate(self.ul_test_loader):
                if torch.cuda.is_available():
                    images = images.cuda()
                outputs = model(images)
                ul_preds[inds, :] = outputs.softmax(dim=1).detach().data.cpu()
        model.train()

        return ul_preds

     
    def _rebuild_labeled_dataset(self):
        
        per_class_sample = self.current_label_dist().tolist()
        self.logger.info("old distributions of labeled dataset:")
        self.logger.info(per_class_sample)
        self.logger.info(
            "imb ratio: {:.2f}".format(
                per_class_sample[0] / per_class_sample[self.num_classes - 1]
            )
        )
        
        ul_preds = self.eval_ul_dataset()
        conf, pred_class = torch.max(ul_preds, dim=1)

        selected_inds = []
        selected_labels = []
        for i in range(self.num_classes):
            inds = torch.where(pred_class == i)[0]
            if len(inds) == 0:
                continue
            num_selected = int(self.mu_per_cls[self.num_classes - (i + 1)] * len(inds))
            if num_selected < 1:
                continue

            sorted_inds = torch.argsort(conf[inds], descending=True)
            selected = inds[sorted_inds[:num_selected]]

            selected_inds.extend(selected.tolist())
            selected_labels.extend([i] * num_selected)

        ul_dataset = self.unlabeled_trainloader.dataset
        ul_data_np = ul_dataset.select_dataset(indices=selected_inds, labels=selected_labels)

        new_data_dict = {
            k: np.concatenate([self.init_l_data[k], ul_data_np[k]], axis=0)
            for k in self.init_l_data.keys()
        }
        new_l_dataset = BaseNumpyDataset(new_data_dict, transforms=self.l_transforms,num_classes=self.num_classes)
        new_loader = _build_loader(self.cfg, new_l_dataset)

        self.current_l_dataset = new_l_dataset
        self._l_iter = iter(new_loader)

        # for logging
        per_class_sample = self.current_label_dist().tolist()
        self.logger.info("new distributions of labeled dataset:")
        self.logger.info(per_class_sample)
        self.logger.info(
            "imb ratio: {:.2f}".format(
                per_class_sample[0] / per_class_sample[self.num_classes - 1]
            )
        )