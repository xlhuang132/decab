from yacs.config import CfgNode
from torchvision import transforms
from .transforms import Augmentation, GeneralizedSSLTransform
import numpy as np
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2 

def build_contrastive_transforms(cfg=None): 
    ### color augmentation ###   
    color_jitter_strength=0.5 
    color_jitter = transforms.ColorJitter(0.8*color_jitter_strength, 0.8*color_jitter_strength, 0.8*color_jitter_strength, 0.2*color_jitter_strength)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    transform=transforms.Compose([
            rnd_color_jitter,
            rnd_gray,
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),])
    return TransformTwice(transform)

def build_transforms(cfg: CfgNode, dataset: str) -> tuple:
    algo_name = cfg.ALGORITHM.NAME
    with_unlabeled = algo_name != "Supervised"

    strong_aug = cfg.DATASET.TRANSFORM.STRONG_AUG
    resolution = cfg.DATASET.RESOLUTION

    aug = Augmentation
    if dataset == "cifar10":
        img_size = (32, 32)
        norm_params = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))

    elif dataset == "cifar100":
        norm_params = dict(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
        img_size = (32, 32)

    elif dataset == "stl10":
        img_size = (96, 96)  # original image size
        norm_params = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))

    remix_algo_names = ["ReMixMatch", "ReMixMatchDASO", "ReMixMatchCReST"]
    with_strong_aug = False
    if algo_name in remix_algo_names:
        with_strong_aug = cfg.ALGORITHM.REMIXMATCH.LABELED_STRONG_AUG
   
    l_train = aug(
            cfg, img_size, strong_aug=with_strong_aug, norm_params=norm_params, resolution=resolution
        )

    ul_train = None
    if with_unlabeled:
        if algo_name == "MixMatch":
            # K weak
            ul_train = GeneralizedSSLTransform(
                [
                    aug(cfg, img_size, norm_params=norm_params, resolution=resolution)
                    for _ in range(cfg.ALGORITHM.MIXMATCH.NUM_AUG)
                ]
            )            
        else:
            ul_train = GeneralizedSSLTransform(
                [
                    aug(cfg, img_size, norm_params=norm_params, resolution=resolution),
                    aug(
                        cfg,
                        img_size,
                        strong_aug=strong_aug,
                        norm_params=norm_params,
                        resolution=resolution,
                        ra_first=True
                    )
                ]
            )
    eval_aug = Augmentation(
        cfg,
        img_size,
        flip=False,
        crop=False,
        norm_params=norm_params,
        is_train=False,
        resolution=resolution
    )
    if algo_name == "DARP_ESTIM":
        # for darp estimation stage, unlabeled images are used for 
        # 'evaluating' the confusion matrix
        ul_train = eval_aug
    if algo_name == "MOOD":
        l_train=ul_train
    return l_train, ul_train, eval_aug

 