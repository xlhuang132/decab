
from .base_trainer import BaseTrainer
from .daso_trainer import DASOTrainer
from .fixmatch_trainer import FixMatchTrainer
from .mixmatch_trainer import MixMatchTrainer
from .crest_trainer import CReSTTrainer
from .openmatch_trainer import OpenMatchTrainer
from .decab_trainer import DeCABTrainer 

def build_trainer(cfg):
    alg=cfg.ALGORITHM.NAME
    if alg=='FixMatch':
        return FixMatchTrainer(cfg) 
    elif alg=='MixMatch':
        return MixMatchTrainer(cfg)
    elif alg=='CReST':
        return CReSTTrainer(cfg)
    elif alg=='DASO':
        return DASOTrainer(cfg) 
    elif alg=='OpenMatch':
        return OpenMatchTrainer(cfg)  
    elif alg== 'DeCAB':
        return DeCABTrainer(cfg)      
    else:
        raise "The algorithm type is not valid!"