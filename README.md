# EEC: Learning to Encode and Regenerate Images for Continual Learning
Pytorch code for the paper: [EEC: Learning to Encode and Regenerate Images for Continual Learning](https://arxiv.org/pdf/2101.04904.pdf)
## Abstract
The two main impediments to continual learning are catastrophic forgetting and memory limitations on the storage of data. To cope with these challenges, we propose a novel, cognitively-inspired approach which trains autoencoders with Neural Style Transfer to encode and store images. During training on a new task, reconstructed images from encoded episodes are replayed in order to avoid catastrophic forgetting. The loss function for the reconstructed images is weighted to reduce its effect during classifier training to cope with image degradation. When the system runs out of memory the encoded episodes are converted into centroids and covariance matrices, which are used to generate pseudo-images during classifier training, keeping classifier performance stable while using less memory. Our approach increases classification accuracy by 13-17% over state-of-the-art methods on benchmark datasets, while requiring 78% less storage space.  

## Applied on ImageNet-50, CIFAR-10, CIFAR-100, MNIST and SVHN 

### Requirements
* torch (Currently working with 1.3.1)
* Scipy (Currently working with 1.2.1)
* Scikit Learn (Currently working with 0.21.2)
* You can use requirements.txt to install the required libraries
* Download the datasets in */data directory
## Usage
* Run ```multiple_auto_decay.py``` to run EEC with multiple autoencoders without using pseudorehearsal.
* Run ```multiple_pseudo.py``` to run EEC with multiple autoencoders with pseudorehearsal.
* The code currently has parameters set for ImageNet-50. Just change the appropriate parameters to run it on other datasets.
## If you consider citing us
```
@inproceedings{
ayub2021eec,
title={{\{}EEC{\}}: Learning to Encode and Regenerate Images for Continual Learning},
author={Ali Ayub and Alan Wagner},
booktitle={International Conference on Learning Representations},
year={2021}
}
```
