
import torch.nn as nn  
from .semiloss import SemiLoss 
import torch.nn as nn
from yacs.config import CfgNode 
from typing import List

from .ce_loss import build_ce_loss  
from .mse_loss import build_mse_loss

def build_loss(
    cfg, **kwargs
) :
    if "combine" in kwargs:
        criterion,val_criterion=None,None
        if cfg.MODEL.LOSS.LABELED_LOSS =="MixmatchLoss": 
            criterion=SemiLoss()
        elif cfg.MODEL.LOSS.LABELED_LOSS =="SemiLoss": 
            criterion=SemiLoss()
        else:
            criterion=nn.CrossEntropyLoss()
        val_criterion=nn.CrossEntropyLoss()
        return criterion,val_criterion
    else:
        l_criterion,ul_criterion,val_criterion=None,None,None
    
        if cfg.MODEL.LOSS.LABELED_LOSS =="CrossEntropyLoss":
            l_criterion=build_ce_loss(cfg,  **kwargs) 
        else:
            raise "Train l_loss{} is not valid!".format(cfg.MODEL.LOSS.LABELED_LOSS)

        if cfg.MODEL.LOSS.UNLABELED_LOSS =="CrossEntropyLoss":
            ul_criterion=build_ce_loss(cfg,  **kwargs)
        elif cfg.MODEL.LOSS.UNLABELED_LOSS =="MSELoss":
            ul_criterion=build_mse_loss(cfg,  **kwargs)
            
        else:
            raise "Train ul_loss{} is not valid!".format(cfg.MODEL.LOSS.UNLABELED_LOSS)

        val_criterion=nn.CrossEntropyLoss()
        return l_criterion,ul_criterion,val_criterion
        