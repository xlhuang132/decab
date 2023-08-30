# DeCAB: Debiased Semi-Supervised Learning for Imbalanced Open-Set Data
**Abstract** : Semi-supervised learning (SSL) has received significant attention due to its ability to use limited labeled data and various unlabeled data to train models with high generalization performance. However, the assumption of a balanced class distribution in traditional SSL approaches limits a wide range of real applications, where the training data exhibits long-tailed distributions. As a consequence, the model is biased towards head classes and disregards tail classes, thereby leading to severe class-aware bias. Additionally, since the unlabeled data may contain out-of-distribution (OOD) samples without manual filtering, the model will be inclined to assign OOD samples to non-tail classes with high confidence, which further overwhelms the tail classes. To alleviate this class-aware bias, we propose an end-to-end semi-supervised method \textit{De}bias \textit{C}lass-\textit{A}ware \textit{B}ias (DeCAB). DeCAB introduces positive-pair scores for contrastive learning instead of positive-negative pairs based on unreliable pseudo-labels, avoiding false negative pairs negatively impacts the feature space. At the same time, DeCAB utilizes class-aware thresholds to select more tail samples and selective sample reweighting for feature learning, preventing OOD samples from being misclassified as head classes and accelerating the convergence speed of the model. Experimental results demonstrate that DeCAB is robust in various semi-supervised benchmarks and achieves state-of-the-art performance.
#### Dependencies
- python 3.7.12
- PyTorch 1.8.1
- torchvision 0.9.1
- CUDA 11.1
- cuDNN 8.0.4
#### Dataset
- CIFAR-10
- CIFAR-100
- Tiny ImageNet
- LSUN
#### Usage
Here is an example to run DeCAB on CIFAR-10:

`python train.py --cfg cfg/cifar10_decab.yaml`

