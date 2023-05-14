
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
from utils import interleave,linear_rampup
from loss.build_loss import build_loss  
from utils import FusionMatrix

class MixMatchTrainer(BaseTrainer):    
    
    def build_loss(self):
        self.criterion,self.val_criterion = build_loss(cfg,combine=True)
        
    def train_step(self,pretraining=False):
        self.model.train()
        loss =0 
        try:
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader)
            inputs_x, targets_x,_ = self.labeled_train_iter.next() 
        batch_size=inputs_x.size(0)
         
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
        
        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = self.model(inputs_u)
            outputs_u2 = self.model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/self.cfg.ALGORITHM.MIXMATCH.TEMPERATURE)  
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()
            
        # mixup
        bin_labels = F.one_hot(targets_x, num_classes=targets_u.size(1)).float()
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([bin_labels, targets_u, targets_u], dim=0)

        l = np.random.beta(self.cfg.ALGORITHM.MIXMATCH.MIXUP_ALPHA, self.cfg.ALGORITHM.MIXMATCH.MIXUP_ALPHA) 

        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
        
        
        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size)) # 按bs把数据切分为多片
        mixed_input = interleave(mixed_input, batch_size)

        logits = [self.model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(self.model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)

        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
    
        lx, lu, w = self.criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], self.epoch+self.iter/self.step_per_epoch)
        # compute 1st branch accuracy
        score_result = self.func(logits_x)
        now_result = torch.argmax(score_result, 1)         
        loss+=lx+lu
        

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        
        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(lx.item(), inputs_x.size(0))
        self.losses_u.update(lu.item(), inputs_u.size(0)) 
        
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='.format(self.epoch,self.iter%self.step_per_epoch if self.iter%self.step_per_epoch>0 else self.step_per_epoch,self.step_per_epoch,self.losses.val,self.losses_x.avg,self.losses_u.val))
        
        return now_result.cpu().numpy(), targets_x.cpu().numpy()
