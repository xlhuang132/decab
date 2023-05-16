
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
from .base_trainer import BaseTrainer  
from utils import FusionMatrix
from models.projector import  Projector 
class FixMatchTrainer(BaseTrainer):    
     
    def train_step(self,pretraining=False): 
        self.model.train()
        loss =0 
        try:        
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader)
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
            
        if  isinstance(inputs_x,list)  :
            inputs_x=inputs_x[0]
          
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
        lx=self.l_criterion(logits_x, targets_x.long()) 
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
    
    