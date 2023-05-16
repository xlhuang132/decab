from .cifar10 import *
from .cifar100 import * 
from .build_transform import *
from dataset.ood_dataset_map import ood_dataset_map 
import os

def build_test_dataset(cfg):
    dataset_name=cfg.DATASET.NAME
    root=cfg.DATASET.ROOT
    _,_,transform_val=build_transform(cfg)
    if dataset_name=='cifar10':
        test_dataset=get_cifar10_test_dataset(root,transform_val=transform_val)
        
    else:
        raise "Dataset name {} is not valid!".format(dataset_name)
    print("Test data distribution:"+str(test_dataset.num_per_cls_list))
    return test_dataset

def build_dataset(cfg,logger=None,test_mode=False):
    dataset_name=cfg.DATASET.NAME
    dataset_root=cfg.DATASET.ROOT 
    ood_dataset=ood_dataset_map[cfg.DATASET.DU.OOD.DATASET] if cfg.DATASET.DU.OOD.ENABLE else 'None'
    transform_train,transform_train_ul,transform_val=build_transform(cfg)
    if dataset_name=='cifar10':
        datasets=get_cifar10(dataset_root,  ood_dataset,
                 transform_train=transform_train,
                 transform_train_ul=transform_train_ul, transform_val=transform_val,
                 download=True,cfg=cfg,logger=logger,test_mode=test_mode)
    elif dataset_name=='cifar100':
        datasets=get_cifar100(dataset_root,  ood_dataset,
                 transform_train=transform_train, transform_train_ul=transform_train_ul, transform_val=transform_val,
                 download=True,cfg=cfg,logger=logger,test_mode=test_mode)
    else:
        raise "Dataset is not valid!"
    
    return datasets

