import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, model_name=None,expansion=4,alg_name=''):
        super(Projector, self).__init__()
        if model_name: 
            if model_name == 'WRN_28_2':
                self.linear_1 = nn.Linear(128, 128)
                self.linear_2 = nn.Linear(128, 64)
            elif model_name == 'WRN_28_8':
                self.linear_1 = nn.Linear(512, 128)
                self.linear_2 = nn.Linear(128, 64)
        elif expansion == 0:
            self.linear_1 = nn.Linear(128, 128)
            self.linear_2 = nn.Linear(128, 64)
        else:
            self.linear_1 = nn.Linear(512*expansion, 512)
            self.linear_2 = nn.Linear(512, 64)
     
    def forward(self, x, internal_output_list=False,normalized=False):
            
        output_list = []

        output = self.linear_1(x)
        output = F.relu(output)
        
        output_list.append(output)
        if normalized: # l2norm
            output = F.normalize(self.linear_2(output),dim=-1)
        else:
            output = self.linear_2(output)
        output_list.append(output)
        
        if internal_output_list:
            return output,output_list
        return output 