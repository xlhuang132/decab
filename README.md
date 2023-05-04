# decab
**Abstract** :Semi-supervised learning (SSL) has received significant attention due to its ability to use limited labeled data and various unlabeled data to train models with high generalization performance. However, common SSL methods assume that target distribution is balanced and there is no out-of-distribution (OOD) data in unlabled data, which is too harsh. In reality, data are frequently distributed with long tails, causing the model to be biased towards the head classes and ignoring the tail classes, resulting in more serious confirmation bias. In addition, since there may be OOD samples in the unlabeled data, in the case of intensified confirmation bias, the model is more inclined to assign OOD samples to non-tail classes with high confidence, resulting in more overwhelming for tail classes. To alleviate the class-aware bias problems, we propose Debiases Class-Aware Bias (DeCAB), which is an end-to-end semi-supervised method that incorporates contrastive information. Considering the class-aware bias of the model in real-world scenarios, its pseudo-labels are quite unreliable, so we introduce positive-pair scores instead of positive-negative pairs based on pseudo-labels for contrastive learning.
Our method demonstrates robustness in various semi-supervised benchmarks and achieves state-of-the-art performance.
#### Dependencies
- python 3.7.12
- PyTorch 1.8.1
- torchvision 0.9.1
- CUDA 11.1
- cuDNN 8.0.4
#### Dataset
- CIFAR-10
- CIFAR-100
- Tine ImageNet
- LSUN
#### Usage
Here is an example to run DeCAB on CIFAR-10:

`python train.py --cfg cfg/cifar10_decab.yaml`
