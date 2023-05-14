import numpy as np
import torch

def get_class_weight(num_list,class_weight_type=None):
    if class_weight_type=='CBLoss':
        return get_class_weight_effective_number(num_list)
    elif class_weight_type=='Reweight':
        return get_class_weight_reversed_number(num_list)
    else:
        return torch.ones(len(num_list)).float()
    
def get_class_weight_effective_number(num_list,beta= 0.9999): # 有效数重加权
    num_classes=len(num_list)
    effective_num = 1. - np.power(beta, num_list)
    effective_num = np.array(effective_num)
    effective_num[effective_num == 1] = np.inf
    per_cls_weights = (1. - beta) / effective_num
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * num_classes
    return torch.tensor(per_cls_weights).float()

def get_class_weight_reversed_number(num_list):
    num_classes=len(num_list)
    per_cls_weights = 1 / np.array(num_list)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * num_classes
    return torch.tensor(per_cls_weights).float()