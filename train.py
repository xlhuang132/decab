 
import torch 
import argparse
from config.defaults import update_config,_C as cfg
from trainer.build_trainer import build_trainer 
import random
import os
import numpy as np
import torch.backends.cudnn as cudnn   
def parse_args():
    parser = argparse.ArgumentParser(description="codes for DeCAB")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="cfg/decab_cifar10.yaml",
        type=str,
    ) 
    
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args 

 
seed=7
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True 
np.random.seed(seed)
args = parse_args()
update_config(cfg, args)  
trainer=build_trainer(cfg)
trainer.train()
 
        
        
    
   