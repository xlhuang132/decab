import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import numpy as np
class RandomSampler(Sampler):
    """ sampling without replacement """

    def __init__(self, data_source, total_samples=None, shuffle=True):
        try:
            data_size = len(data_source.dataset["images"] ) 
        except:
            data_size=len(data_source)
        # total train epochs
        num_epochs = total_samples // data_size + 1
        _indices = torch.cat([torch.randperm(data_size) for _ in range(num_epochs)])
        self._indices = _indices.tolist()[:total_samples]     
        
    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)
 