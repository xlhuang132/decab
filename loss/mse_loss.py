"""Reference: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/cross_entropy_loss.py"""  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from yacs.config import CfgNode

from typing import List, Optional, Tuple 
from .utils import weight_reduce_loss, weighted_loss


def build_mse_loss(cfg: CfgNode, loss_weight: float = 1.0, class_weight=None,use_sigmoid=False): 
    return MSELoss(
        use_sigmoid=use_sigmoid, loss_weight=loss_weight, class_weight=class_weight
    )
    
@weighted_loss
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Warpper of mse loss."""
    return F.mse_loss(pred, target, reduction='none')


class MSELoss(nn.Module):
    def __init__(
        self,
        use_sigmoid: bool = False,
        reduction: str = 'mean',
        class_weight: Optional[List[float]] = None,
        loss_weight: Optional[float] = 1.0
    ) :
        """MSELoss.
        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(MSELoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.cls_criterion = mse_loss

    def forward(
        self,
        cls_score: Tensor,
        label: Tensor,
        *,
        with_activation: bool = False,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
        class_weight_override=None,
        **kwargs
    ) :
        """Forward function.
        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:  
            class_weight = self.class_weight
            if isinstance(class_weight, list):
                class_weight = cls_score.new_tensor(class_weight)
        else:
            class_weight = None
        if class_weight_override is not None:
            class_weight = class_weight_override
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            **kwargs
        ) 
        return loss_cls
