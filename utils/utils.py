import numpy as np
import torch
import os
import copy
import sys
from scipy import misc  
 
def linear_rampup(current, rampup_length=16):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
    
def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1  
    offsets = interleave_offsets(batch, nu)   
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]  
    for i in range(1, nu + 1):  
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]   
    return [torch.cat(v, dim=0) for v in xy] 
 
def get_group_splits(cfg):  
    num_classes=cfg.DATASET.NUM_CLASSES
    group_splits=cfg.DATASET.GROUP_SPLITS
    assert num_classes==sum(group_splits)
    id_splits=[]
    l=0
    class_ids=np.array([i for i in range(num_classes)])
    for item in group_splits:
        id_splits.append(class_ids[l:l+item])
        l+=item
    return id_splits
  
def get_DL_dataset_path(cfg,dataset=None,):
    dataset = cfg.DATASET.NAME if not dataset else dataset
    path=os.path.join(cfg.OUTPUT_DIR, dataset)
    return path

def get_DL_dataset_alg_path(cfg,dataset=None,algorithm=None,labeled_loss_type=None):
    parent_path=get_DL_dataset_path(cfg,dataset=dataset)
    algorithm_name=cfg.ALGORITHM.NAME   if not algorithm else algorithm 
    model_name=cfg.MODEL.NAME
    labeled_loss_type=cfg.MODEL.LOSS.LABELED_LOSS_CLASS_WEIGHT_TYPE if not labeled_loss_type else labeled_loss_type
    if labeled_loss_type and labeled_loss_type!='None':
        algorithm_name=algorithm_name+labeled_loss_type
    path=os.path.join(parent_path, algorithm_name, model_name)
    return path

def get_DL_dataset_alg_DU_dataset_path(cfg,dataset=None,algorithm=None,labeled_loss_type=None,
             num_labeled_head=None,imb_factor_l=None,num_unlabeled_head=None,imb_factor_ul=None):
    parent_path=get_DL_dataset_alg_path(cfg,dataset=dataset,algorithm=algorithm,labeled_loss_type=labeled_loss_type)
    num_labeled_head=cfg.DATASET.DL.NUM_LABELED_HEAD if not num_labeled_head else num_labeled_head
    imb_factor_l=cfg.DATASET.DL.IMB_FACTOR_L if not imb_factor_l else imb_factor_l
    num_unlabeled_head=cfg.DATASET.DU.ID.NUM_UNLABELED_HEAD if not num_unlabeled_head else num_unlabeled_head
    imb_factor_ul=cfg.DATASET.DU.ID.IMB_FACTOR_UL  if not imb_factor_ul else imb_factor_ul
 
    DL_DU_ID_setting='DL-{}-IF-{}-DU{}-IF_U-{}'.format(num_labeled_head,imb_factor_l,num_unlabeled_head,imb_factor_ul)
    path=os.path.join(parent_path, DL_DU_ID_setting)
    return path

def get_DL_dataset_alg_DU_dataset_OOD_path(cfg,dataset=None,algorithm=None,labeled_loss_type=None,
             num_labeled_head=None,imb_factor_l=None,num_unlabeled_head=None,imb_factor_ul=None,
             ood_dataset=None): 
    parent_path=get_DL_dataset_alg_DU_dataset_path(cfg,dataset=dataset,algorithm=algorithm,labeled_loss_type=labeled_loss_type,
             num_labeled_head=num_labeled_head,imb_factor_l=imb_factor_l,num_unlabeled_head=num_unlabeled_head,imb_factor_ul=imb_factor_ul)
    ood_dataset=cfg.DATASET.DU.OOD.DATASET if not ood_dataset else ood_dataset 
    OOD_setting='OOD-{}-all'.format(ood_dataset)
    path=os.path.join(parent_path, OOD_setting)
    return path

def get_root_path(cfg,dataset=None,algorithm=None,labeled_loss_type=None,
             num_labeled_head=None,imb_factor_l=None,num_unlabeled_head=None,imb_factor_ul=None,
             ood_dataset=None,
             sampler=None,sampler_mixup=None,dual_sampler_enable=None,dual_sampler=None,dual_sampler_mixup=None,
             Branch_setting=None,  
             ):
      
    path=get_DL_dataset_alg_DU_dataset_OOD_path(cfg,dataset=dataset,algorithm=algorithm,labeled_loss_type=labeled_loss_type,
             num_labeled_head=num_labeled_head,imb_factor_l=imb_factor_l,
             num_unlabeled_head=num_unlabeled_head,imb_factor_ul=imb_factor_ul,
             ood_dataset=ood_dataset
             ) 
    return path 

def prepare_output_path(cfg,logger): 
    path= get_root_path(cfg)
    model_dir = os.path.join(path ,"models") 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        logger.info(
            "This directory has already existed, Please remember to modify your cfg.NAME"
        ) 
    print("=> output model will be saved in {}".format(model_dir)) 
    pic_dir= os.path.join(path ,"pic") 
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir) 
    return path,model_dir,pic_dir 
  

 