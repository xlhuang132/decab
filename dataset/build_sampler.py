
 
from .random_sampler import * 
 

def build_sampler(cfg,dataset,sampler_type="RandomSampler",total_samples=None):
    assert sampler_type!=None and total_samples!=None
    if sampler_type == "RandomSampler": 
        sampler = RandomSampler(dataset,total_samples=total_samples)  
    else:
        raise ValueError     
    return sampler 