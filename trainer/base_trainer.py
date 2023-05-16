   
import logging
from operator import mod
from tkinter import W
import torch 
import numpy as np
from dataset.build_dataloader import *
from loss.build_loss import build_loss 
import models 
import time 
import torch.optim as optim
from models.feature_queue import FeatureQueue
import os   
import datetime 
from utils.utils import *
import torch.nn.functional as F 
from utils import AverageMeter, create_logger,prepare_output_path
from utils.build_optimizer import get_optimizer, get_scheduler 
from dataset.build_dataloader import _build_loader
from dataset.base import BaseNumpyDataset
from utils import FusionMatrix 
from models.projector import  Projector 
 
class BaseTrainer():
    def __init__(self,cfg):
        self.local_rank=cfg.LOCAL_RANK
        self.cfg=cfg
        self.logger, _ = self.create_logger(cfg)        
        self.path,self.model_dir,self.pic_dir =self.prepare_output_path(cfg,self.logger)
        self.num_classes=cfg.DATASET.NUM_CLASSES
        self.batch_size=cfg.DATASET.BATCH_SIZE
        # =================== build model ============= 
        self.model = models.__dict__[cfg.MODEL.NAME](cfg=cfg) 
        self.model=self.model.cuda()
        # =================== build dataloader =============
        self.ul_test_loader=None
        self.build_data_loaders()
        # =================== build criterion ==============
        self.build_loss()
        # ========== build optimizer ===========         
        self.optimizer = get_optimizer(cfg, self.model)
        
        # ========== build dataloader ==========     
        
        self.max_epoch=cfg.MAX_EPOCH 
        self.step_per_epoch=cfg.TRAIN_STEP   
        self.max_iter=self.max_epoch*self.step_per_epoch+1
        self.func = torch.nn.Softmax(dim=1)  
        self.conf_thres=cfg.ALGORITHM.CONFIDENCE_THRESHOLD   
        
        self.iter=0
        self.best_val=0
        self.best_val_iter=0
        self.best_val_test=0
        self.start_iter=1
        self.epoch=1
        self.save_epoch=cfg.SAVE_EPOCH
        
        
        self.l_num=len(self.labeled_trainloader.dataset)
        self.ul_num=len(self.unlabeled_trainloader.dataset)   
        self.feature_dim=64   
        self.opearte_before_resume()
         
    def prepare_output_path(self,cfg,logger):
        return prepare_output_path(cfg,logger)
    
    def create_logger(self,cfg) :
        return create_logger(cfg) 
    
    def opearte_before_resume(self):
        pass     
    
    @classmethod
    def build_model(cls, cfg)  :
        model = models.__dict__[cfg.MODEL.NAME](cfg)
        return model
    
    @classmethod
    def build_optimizer(cls, cfg , model )  :
        return get_optimizer(cfg, model)
    
    @classmethod
    def build_scheduler(cls, cfg , optimizer )  :
        return get_scheduler(cfg, optimizer)
    
    def build_loss(self):
        self.l_criterion,self.ul_criterion,self.val_criterion = build_loss(self.cfg)
        return 
    
    def build_data_loaders(self,)  :
         
        dataloaders=build_dataloader(self.cfg,self.logger)
        
        self.domain_trainloader=dataloaders[0]
        self.labeled_trainloader=dataloaders[1]
        self.labeled_train_iter=iter(self.labeled_trainloader)        
        # DU               
        self.unlabeled_trainloader=dataloaders[2]
        self.unlabeled_train_iter=iter(self.unlabeled_trainloader)   
        self.val_loader=dataloaders[3]
        self.test_loader=dataloaders[4]
        self.pre_train_loader=dataloaders[5] 
        self.pre_train_iter=iter(self.pre_train_loader)  
        return  
    
    def train_step(self,pretraining=False):
        pass
    
    def loss_init(self):
        self.losses = AverageMeter()
        self.losses_x = AverageMeter()
        self.losses_u = AverageMeter() 
        
    def train(self,):
        fusion_matrix = FusionMatrix(self.num_classes)
        acc = AverageMeter()      
        self.loss_init()
        start_time = time.time()   
        for self.iter in range(self.start_iter, self.max_iter): 
            return_data=self.train_step()
            if return_data is not None:
                pred,gt=return_data[0],return_data[1]
                fusion_matrix.update(pred, gt) 
            if self.iter%self.step_per_epoch==0:  
                end_time = time.time()           
                time_second=(end_time - start_time)
                eta_seconds = time_second * (self.max_epoch - self.epoch)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    
                group_acc=fusion_matrix.get_group_acc(self.cfg.DATASET.GROUP_SPLITS) 
                results=self.evaluate()
                
                if self.best_val<results[0]:
                    self.best_val=results[0]
                    self.best_val_test=results[1]
                    self.best_val_iter=self.iter
                    self.save_checkpoint(file_name="best_model.pth")
                    
                if self.epoch%self.save_epoch==0:
                    self.save_checkpoint()
                self.train_losses.append(self.losses.avg)
                self.logger.info("== Pretraining is enable:{}".format(self.pretraining))
                self.logger.info('== Train_loss:{:>5.4f}  train_loss_x:{:>5.4f}   train_loss_u:{:>5.4f} '.\
                    format(self.losses.avg, self.losses_x.avg, self.losses_u.avg))
                self.logger.info('== val_losss:{:>5.4f}   test_loss:{:>5.4f}   epoch_Time:{:>5.2f}min eta:{}'.\
                        format(self.val_losses[-1], self.test_losses[-1],time_second / 60,eta_string))
                self.logger.info('== Train  group_acc: many:{:>5.2f}  medium:{:>5.2f}  few:{:>5.2f}'.format(self.train_group_accs[-1][0]*100,self.train_group_accs[-1][1]*100,self.train_group_accs[-1][2]*100))
                self.logger.info('==  Val   group_acc: many:{:>5.2f}  medium:{:>5.2f}  few:{:>5.2f}'.format(self.val_group_accs[-1][0]*100,self.val_group_accs[-1][1]*100,self.val_group_accs[-1][2]*100))
                self.logger.info('==  Test  group_acc: many:{:>5.2f}  medium:{:>5.2f}  few:{:>5.2f}'.format(self.test_group_accs[-1][0]*100,self.test_group_accs[-1][1]*100,self.test_group_accs[-1][2]*100))
                self.logger.info('== Val_acc:{:>5.2f}  Test_acc:{:>5.2f}'.format(results[0]*100,results[1]*100))
                self.logger.info('== Best Results: Epoch:{} Val_acc:{:>5.2f}  Test_acc:{:>5.2f}'.format(self.best_val_iter//self.step_per_epoch,self.best_val*100,self.best_val_test*100))
              
                # reset 
                fusion_matrix = FusionMatrix(self.num_classes)
                acc = AverageMeter()                 
                self.loss_init()             
                start_time = time.time()   
                self.operate_after_epoch()
                self.epoch+=1   
                
        self.plot()       
        return
    
    def build_data_loaders_for_dl_contra(self,)  :  
        l_dataset = self.labeled_trainloader.dataset
        l_data_np= l_dataset.select_dataset()
        _,transform= self.pre_train_loader.dataset.select_dataset(return_transforms=True)
        new_l_dataset = BaseNumpyDataset(l_data_np, transforms=transform,num_classes=self.num_classes)
        new_loader = _build_loader(self.cfg, new_l_dataset,is_train=False)
        self.pre_train_loader=new_loader
        self.pre_train_iter=iter(self.pre_train_loader)
        return  
    
    def rebuild_unlabeled_dataset(self,selected_inds):
        ul_dataset = self.unlabeled_trainloader.dataset
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
        self.logger.info("=="*30) 
    
    def get_val_model(self,):
        return self.model
    
    def _rebuild_models(self):
        model = self.build_model(self.cfg) 
        self.model = model.cuda() 
        
    def _rebuild_optimizer(self, model):
        self.optimizer = self.build_optimizer(self.cfg, model) 
        torch.cuda.empty_cache() 
    
    def get_test_best(self):
        return self.best_val_test
    
    def evaluate(self,return_group_acc=False,return_class_acc=False):  
        eval_model=self.get_val_model() 
        test_loss, test_acc ,test_group_acc,test_class_acc=  self.eval_loop(eval_model,self.test_loader, self.val_criterion)
        if self.valset_enable:
            val_loss, val_acc,val_group_acc,val_class_acc = self.eval_loop(eval_model,self.val_loader, self.val_criterion) 
        else: 
            val_loss, val_acc,val_group_acc,val_class_acc=test_loss, test_acc ,test_group_acc,test_class_acc
        
        if return_group_acc:
            if return_class_acc:
                return val_acc,test_acc,test_group_acc,test_class_acc
            else:
                return val_acc,test_acc,test_group_acc
        if return_class_acc:
            return val_acc,test_acc,test_class_acc
        return [val_acc,test_acc]
     
    def eval_loop(self,model,valloader,criterion):
        losses = AverageMeter()  
        model.eval()
 
        fusion_matrix = FusionMatrix(self.num_classes)
        func = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            for  i, (inputs, targets, _) in enumerate(valloader):
                # measure data loading time 

                inputs, targets = inputs.cuda(), targets.long().cuda(non_blocking=True)

                # compute output
                outputs = model(inputs)
                if len(outputs)==2 and len(outputs)!=len(targets):
                    outputs=outputs[0]
                loss = criterion(outputs, targets)

                # measure accuracy and record loss 
                losses.update(loss.item(), inputs.size(0)) 
                score_result = func(outputs)
                now_result = torch.argmax(score_result, 1) 
                fusion_matrix.update(now_result.cpu().numpy(), targets.cpu().numpy())
                 
        group_acc=fusion_matrix.get_group_acc(self.cfg.DATASET.GROUP_SPLITS)
        class_acc=fusion_matrix.get_acc_per_class()
        acc=fusion_matrix.get_accuracy()    
        return (losses.avg, acc, group_acc,class_acc)
  
    def save_checkpoint(self,file_name=""):
        if file_name=="":
            file_name="checkpoint.pth" if self.iter!=self.max_iter else "model_final.pth"
        torch.save({
                    'model': self.model.state_dict(), 
                    'iter': self.iter, 
                    'best_val': self.best_val, 
                    'best_val_iter':self.best_val_iter, 
                    'best_val_test': self.best_val_test,
                    'optimizer': self.optimizer.state_dict(),                          
                },  os.path.join(self.model_dir, file_name))
        return 
    
    def load_checkpoint(self, resume) :
        self.logger.info(f"resume checkpoint from: {resume}")

        state_dict = torch.load(resume)
        # load model 
        model_dict=self.model.state_dict() 
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict["model"].items() if k in self.model.state_dict()}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict) 

        # load optimizer 
        try:
            self.optimizer.load_state_dict(state_dict["optimizer"])  
        except: 
            self.logger.warning('load optimizer wrong!')
        self.start_iter=state_dict["iter"]+1
        self.best_val=state_dict['best_val']
        self.best_val_iter=state_dict['best_val_iter']
        self.best_val_test=state_dict['best_val_test']  
        self.epoch= (self.start_iter // self.step_per_epoch) 
        self.logger.info(
            "Successfully loaded the checkpoint. "
            f"start_iter: {self.start_iter} start_epoch:{self.epoch} " 
        ) 
     
    
    def get_class_counts(self,dataset):
        """
            Sort the class counts by class index in an increasing order
            i.e., List[(2, 60), (0, 30), (1, 10)] -> np.array([30, 10, 60])
        """
        return np.array(dataset.num_per_cls_list) 
    
    def get_label_dist(self, dataset=None, normalize=None):
        """
            normalize: ["sum", "max"]
        """
        if dataset is None:
            dataset = self.labeled_trainloader.dataset

        class_counts = torch.from_numpy(self.get_class_counts(dataset)).float()
        class_counts = class_counts.cuda()

        if normalize:
            assert normalize in ["sum", "max"]
            if normalize == "sum":
                return class_counts / class_counts.sum()
            if normalize == "max":
                return class_counts / class_counts.max()
        return class_counts

    def build_labeled_loss(self, cfg , warmed_up=False)  :
        loss_type = cfg.MODEL.LOSS.LABELED_LOSS
        num_classes = cfg.MODEL.NUM_CLASSES
        assert loss_type == "CrossEntropyLoss"

        class_count = self.get_label_dist(device=self.device)
        per_class_weights = None
        if cfg.MODEL.LOSS.WITH_LABELED_COST_SENSITIVE and warmed_up:
            loss_override = cfg.MODEL.LOSS.COST_SENSITIVE.LOSS_OVERRIDE
            beta = cfg.MODEL.LOSS.COST_SENSITIVE.BETA
            if beta < 1:
                # effective number of samples;
                effective_num = 1.0 - torch.pow(beta, class_count)
                per_class_weights = (1.0 - beta) / effective_num
            else:
                per_class_weights = 1.0 / class_count

            # sum to num_classes
            per_class_weights = per_class_weights / torch.sum(per_class_weights) * num_classes

            if loss_override == "":
                # CE loss
                loss_fn = build_loss(
                    cfg, loss_type, class_count=class_count, class_weight=per_class_weights
                ) 
            else:
                raise ValueError()
        else:
            loss_fn = build_loss(
                cfg, loss_type, class_count=class_count, class_weight=per_class_weights
            )

        return loss_fn

   