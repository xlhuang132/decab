
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
from dataset.build_dataloader import _build_loader  
from dataset.base import BaseNumpyDataset
from .base_trainer import BaseTrainer
import copy
from loss.contrastive_loss import *
from utils import FusionMatrix,AverageMeter
from models.projector import  Projector 

def ova_loss(logits_open, label):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.range(0, logits_open.size(0) - 1).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :]
                                                    + 1e-8) * label_sp_neg, 1)[0])
    Lo = open_loss_neg + open_loss
    return Lo


def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open *
                                   torch.log(logits_open + 1e-8), 1), 1))
    return Le


class OpenMatchTrainer(BaseTrainer):   
    def __init__(self, cfg):        
        super().__init__(cfg)       
        self.lambda_oem=cfg.ALGORITHM.OPENMATCH.LAMBDA_OEM
        self.lambda_socr=cfg.ALGORITHM.OPENMATCH.LAMBDA_SOCR
        self.mu=cfg.ALGORITHM.OPENMATCH.MU
        self.T=cfg.ALGORITHM.OPENMATCH.T
        self.start_fix=cfg.ALGORITHM.OPENMATCH.START_FIX
        self.unlabeled_all_trainloader=copy.deepcopy(self.unlabeled_trainloader)
        self.unlabeled_all_trainloader_iter=iter(self.unlabeled_all_trainloader)
        if self.cfg.RESUME !="":
            self.load_checkpoint(self.cfg.RESUME)  
        
        
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter()  
        self.losses_o = AverageMeter()
        self.losses_oem = AverageMeter()
        self.losses_socr = AverageMeter()
        self.losses_fix = AverageMeter()
        self.mask_probs = AverageMeter()
    
    
    def train_step(self,pretraining=False):
        
        self.model.train()
        loss =0 
        try:        
            (_, inputs_x_s, inputs_x), targets_x,_ = self.labeled_train_iter.next() 
        except:
            self.labeled_train_iter=iter(self.labeled_trainloader)
            (_, inputs_x_s, inputs_x), targets_x,_ = self.labeled_train_iter.next() 
         
        try:       
            data = self.unlabeled_train_iter.next()
        except:
            self.unlabeled_train_iter=iter(self.unlabeled_trainloader)
            data = self.unlabeled_train_iter.next()
        inputs_u_w=data[0][0]
        inputs_u_s=data[0][1] 
         
        try:
            (inputs_all_w, inputs_all_s, _), _, _ = self.unlabeled_all_trainloader_iter.next()
        except:
            self.unlabeled_all_trainloader_iter = iter(self.unlabeled_all_trainloader)
            (inputs_all_w, inputs_all_s, _), _, _ = self.unlabeled_all_trainloader_iter.next()
            
        b_size = inputs_x.shape[0]
        inputs_all = torch.cat([inputs_all_w, inputs_all_s], 0)
        inputs = torch.cat([inputs_x, inputs_x_s,
                            inputs_all], 0).cuda()
        targets_x = targets_x.long().cuda()
     
        logits, logits_open = self.model(inputs)
        logits_open_u1, logits_open_u2 = logits_open[2*b_size:].chunk(2)

        ## Loss for labeled samples
        Lx = F.cross_entropy(logits[:2*b_size],
                                    targets_x.repeat(2), reduction='mean')
        # compute 1st branch accuracy
        score_result = self.func(logits[:2*b_size])
        now_result = torch.argmax(score_result, 1) 
        
        Lo = ova_loss(logits_open[:2*b_size], targets_x.repeat(2))

        ## Open-set entropy minimization
        L_oem = ova_ent(logits_open_u1) / 2.
        L_oem += ova_ent(logits_open_u2) / 2.

        ## Soft consistenty regularization
        logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
        logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
        logits_open_u1 = F.softmax(logits_open_u1, 1)
        logits_open_u2 = F.softmax(logits_open_u2, 1)
        L_socr = torch.mean(torch.sum(torch.sum(torch.abs(
            logits_open_u1 - logits_open_u2)**2, 1), 1))

        if self.epoch >= self.start_fix:
            inputs_ws = torch.cat([inputs_u_w, inputs_u_s], 0).cuda()
            logits, logits_open_fix = self.model(inputs_ws)
            logits_u_w, logits_u_s = logits.chunk(2)
            pseudo_label = torch.softmax(logits_u_w.detach()/self.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.conf_thres).float()
            L_fix = (F.cross_entropy(logits_u_s,
                                        targets_u,
                                        reduction='none') * mask).mean()
            self.mask_probs.update(mask.mean().item())

        else:
            L_fix = torch.zeros(1).cuda().mean()
        loss = Lx + Lo + self.lambda_oem * L_oem  \
                + self.lambda_socr * L_socr + L_fix
          

        # record loss
        self.losses.update(loss.item(), inputs_x.size(0))
        self.losses_x.update(Lx.item(), inputs_x.size(0)) 
        self.losses_o.update(Lo.item())
        self.losses_oem.update(L_oem.item())
        self.losses_socr.update(L_socr.item())
        self.losses_fix.update(L_fix.item())

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        
        if self.iter % self.cfg.SHOW_STEP==0:
            self.logger.info('== Epoch:{} Step:[{}|{}] Total_Avg_loss:{:>5.4f} Avg_Loss_x:{:>5.4f}  Avg_Loss_u:{:>5.4f} =='.format(self.epoch,self.iter%self.step_per_epoch if self.iter%self.step_per_epoch>0 else self.step_per_epoch,self.step_per_epoch,self.losses.avg,self.losses_x.avg,self.losses_u.avg))
            self.logger.info('======= Avg_Lo:{:>5.4f} Avg_L_oem:{:>5.4f}  Avg_L_socr:{:>5.4f}  Avg_L_fix:{:>5.4f} ======='.format(self.losses_o.avg,self.losses_oem.avg,self.losses_socr.avg,self.losses_fix.avg))
         
        return now_result.cpu().numpy(), targets_x.repeat(2).cpu().numpy()
     
    def exclude_dataset(self, exclude_known=False):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, ((_, _, inputs), targets,_) in enumerate(self.unlabeled_all_trainloader):
                inputs = inputs.cuda()
                outputs, outputs_open = self.model(inputs)
                outputs = F.softmax(outputs, 1)
                out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
                tmp_range = torch.range(0, out_open.size(0) - 1).long().cuda()
                pred_close = outputs.data.max(1)[1]
                unk_score = out_open[tmp_range, 0, pred_close]
                known_ind = unk_score < 0.5
                if batch_idx == 0:
                    known_all = known_ind
                else:
                    known_all = torch.cat([known_all, known_ind], 0) 
        known_all = known_all.data.cpu().numpy()
        if exclude_known:
            ind_selected = np.where(known_all == 0)[0]
        else:
            ind_selected = np.where(known_all != 0)[0]
        self.logger.info("selected ratio %s"%( (len(ind_selected)/ len(known_all))))
         
        self.rebuild_unlabeled_dataset(ind_selected)     

    def rebuild_unlabeled_dataset(self,selected_inds):
        ul_dataset = self.unlabeled_all_trainloader.dataset
        ul_data_np,ul_transform = ul_dataset.select_dataset(indices=selected_inds,return_transforms=True)

        new_ul_dataset = BaseNumpyDataset(ul_data_np, transforms=ul_transform,num_classes=self.num_classes)
        new_loader = _build_loader(self.cfg, new_ul_dataset)
        self.unlabeled_trainloader=new_loader
        if self.ul_test_loader is not None:
            _,ul_test_transform=self.ul_test_loader.dataset.select_dataset(return_transforms=True)
            new_ul_test_dataset = BaseNumpyDataset(ul_data_np, transforms=ul_test_transform,num_classes=self.num_classes)
            self.ul_test_loader = _build_loader(
                self.cfg, new_ul_test_dataset, is_train=False, has_label=False
            )
        self.unlabeled_train_iter = iter(new_loader)
    
    def operate_after_epoch(self):  
            if self.epoch >= self.start_fix: 
                self.exclude_dataset()   
            self.loss_init()
            self.logger.info('=='*40)    
            
          