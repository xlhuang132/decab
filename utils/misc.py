'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os 
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from collections import defaultdict

__all__ = [ 'init_params', 'mkdir_p', 'AverageMeter','Meters']

 
def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        

class Meters:
    """Class including a set of AverageMeter"""

    def __init__(self, start_iter: int = 0) -> None:
        self.meters = defaultdict(AverageMeter)
        self._show_avg = {}
        self._iter = start_iter

    def put_scalar(self, name: str, val: float, *, n: int = 1, show_avg: bool = True) -> None:
        self.meters[name].update(val=val, n=n)
        show_average = self._show_avg.get(name)
        if show_average is not None:
            assert show_average == show_avg
        else:
            self._show_avg[name] = show_avg

    def put_scalars(
        self, scalars_dict: dict, *, n: int = 1, show_avg: bool = True, prefix: str = ""
    ) -> None:
        if prefix:
            prefix = prefix + "/"
        for k, v in scalars_dict.items():
            self.put_scalar(name=prefix + k, val=v, show_avg=show_avg, n=n)

    def reset(self) -> None:
        self.meters = defaultdict(AverageMeter)

    def get_latest_scalars_with_avg(self) -> dict:
        result = {}
        for k, meter in self.meters.items():
            result[k] = meter.avg if self._show_avg[k] else meter.val
        return result

    def step(self) -> None:
        self._iter += 1

    @property
    def iter(self) -> int:
        return self._iter