import numpy as np
from PIL import Image
from torch import Tensor
from torchvision import transforms
from yacs.config import CfgNode
 
from typing import Optional, Tuple, Union

from dataset.transform.rand_augment import RandAugment,RandAugmentMC
 
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
class TransformFixMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

class TransformOpenMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.weak(x)

        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(self.weak2(x))
        else:
            return weak, 
class GeneralizedSSLTransform:

    def __init__(self, transforms) :
        assert len(transforms) > 0
        self.transforms = transforms

    def __call__(self, img: Union[np.ndarray, Image.Image]) -> Union[Tensor, Tuple[Tensor]]:
        results = []
        for t in self.transforms:
            results.append(t(img))
        if len(results) == 1:
            return results[0]
        return tuple(results)


class Augmentation:

    def __init__(
        self,
        cfg,
        img_size,
        *,
        flip= True,
        crop = True,
        strong_aug = False,
        norm_params = None,
        is_train= True,
        resolution=32,
        ra_first=False
    ) :
        h, w = img_size
        t = []

        # random horizontal flip
        if flip:
            t.append(transforms.RandomHorizontalFlip())

        # random padding crop
        if crop:
            pad_w = int(w * 0.125) if w == 32 else 4
            pad_h = int(h * 0.125) if h == 32 else 4
            t.append(
                transforms.RandomCrop(img_size, padding=(pad_h, pad_w), padding_mode="reflect")
            )

        if strong_aug and ra_first:
            # apply RA before image resize
            t.append(RandAugment(2, 10, prob=0.5, aug_pool="FixMatch", apply_cutout=True))

        # resize if the actual size of image differs from the desired resolution
        if resolution != h:
            t.append(transforms.Resize((resolution, resolution)))

        if strong_aug and (not ra_first):
            # apply RA after image resize
            t.append(RandAugment(2, 10, prob=0.5, aug_pool="FixMatch", apply_cutout=True))

        # numpy to tensor
        t.append(transforms.ToTensor())

        # normalizer
        if norm_params is not None:
            t.append(transforms.Normalize(**norm_params))

        self.t = transforms.Compose(t)

    def __call__(self, img: Union[np.ndarray, Image.Image]) -> Tensor:
        if isinstance(img, np.ndarray):
            if img.shape[0] == 3:
                img = np.moveaxis(img, 0, -1)
            img = Image.fromarray(img.astype(np.uint8))
        # PIL image type
        assert isinstance(img, Image.Image)
        return self.t(img)


class ContraAugmentation:

    def __init__(
        self,
        cfg,
        img_size,
        *, 
        strong_aug = False,
        norm_params = None,
        is_train= True,
        resolution=32,
        ra_first=False
    ) :
        h, w = img_size
        t = [] 
        
        # random horizontal flip 
        t.append(transforms.RandomHorizontalFlip(p=0.5))

        # random padding crop 
        pad_w = int(w * 0.125) if w == 32 else 4
        pad_h = int(h * 0.125) if h == 32 else 4
        t.append(
            # transforms.RandomResizedCrop(32) 
            transforms.RandomCrop(img_size, padding=(pad_h, pad_w), padding_mode="reflect")
        )
        # color
        t.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        )
        # gray
        t.append(
            transforms.RandomGrayscale(p=0.2),
        ) 
        
        if resolution != h:
            t.append(transforms.Resize((resolution, resolution)))


        # numpy to tensor
        t.append(transforms.ToTensor())

        # normalizer 
        t.append(transforms.Normalize(**norm_params))

        self.t = transforms.Compose(t)

    def __call__(self, img: Union[np.ndarray, Image.Image]) -> Tensor:
        if isinstance(img, np.ndarray):
            if img.shape[0] == 3:
                img = np.moveaxis(img, 0, -1)
            img = Image.fromarray(img.astype(np.uint8))
        # PIL image type
        assert isinstance(img, Image.Image)
        return self.t(img)
