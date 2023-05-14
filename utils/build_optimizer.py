import torch
from utils.lr_scheduler import WarmupMultiStepLR

def get_optimizer(cfg, model):
    """
    Build an optimizer from config.
    """ 
    if cfg.MODEL.OPTIMIZER.TYPE  == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.MODEL.OPTIMIZER.BASE_LR,
            momentum=cfg.MODEL.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.MODEL.OPTIMIZER.WEIGHT_DECAY,
            nesterov=True,
        )
    elif cfg.MODEL.OPTIMIZER.TYPE == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.MODEL.OPTIMIZER.BASE_LR, 
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=cfg.MODEL.OPTIMIZER.WEIGHT_DECAY,
        )
    else:
        raise ValueError("Unknown Optimizer: {}".format(cfg.SOLVER.OPTIM_NAME))
    return optimizer

def get_scheduler(cfg, optimizer):
    if cfg.MODEL.LR_SCHEDULER.TYPE == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            cfg.MODEL.LR_SCHEDULER.LR_STEP,
            gamma=cfg.MODEL.LR_SCHEDULER.LR_FACTOR,
        )
    elif cfg.MODEL.LR_SCHEDULER.TYPE == "cosine":
        if cfg.MODEL.LR_SCHEDULER.COSINE_DECAY_END > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.MODEL.LR_SCHEDULER.COSINE_DECAY_END, eta_min=1e-4
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.MODEL.MAX_EPOCH, eta_min=1e-4
            )
    elif cfg.MODEL.LR_SCHEDULER.TYPE == "warmup":
        scheduler = WarmupMultiStepLR(
            optimizer,
            cfg.MODEL.LR_SCHEDULER.LR_STEP,
            gamma=cfg.MODEL.LR_SCHEDULER.LR_FACTOR,
            warmup_epochs=cfg.MODEL.LR_SCHEDULER.WARM_EPOCH,
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(cfg.MODEL.LR_SCHEDULER.TYPE))

    return scheduler

