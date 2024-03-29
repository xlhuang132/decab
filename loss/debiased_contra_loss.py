""" The Code is under Tencent Youtu Public Rule
Part of the code is adopted form SupContrast as in the comment in the class
"""
from __future__ import print_function

import torch
import torch.nn as nn

from sklearn.metrics.pairwise import cosine_similarity

class DebiasConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, 
            labels=None, 
            mask=None, 
            reduction="mean"):
        
        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases 1-torch.eye
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask # mask self-self

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # compute mean of log-likelihood over positive
        
        sum_mask = mask.sum(1)
        sum_mask[sum_mask == 0] = 1
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / sum_mask
        
       
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)

        if reduction == "mean":
            loss = loss.mean()

        return loss

