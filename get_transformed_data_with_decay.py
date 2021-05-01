import os
import sys
import pickle
import numpy as np

from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset

class getTransformedData(Dataset):
    """transformed dataset for incremental learning
    """
    def __init__(self, images, labels, transform=None, ages = None,seed=1):
        #np.random.seed(seed)
        #torch.manual_seed(seed)
        self.train_images = images
        self.labels = labels
        self.train_images = np.array(self.train_images)
        self.labels = np.array(self.labels)
        #if transform is given, we transoform data using
        self.transform = transform
        self.ages = ages
        if self.ages is not None:
            self.ages = np.array(self.ages)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        #r = self.train_images[index, :1024].reshape(32, 32)
        #g = self.train_images[index, 1024:2048].reshape(32, 32)
        #b = self.train_images[index, 2048:].reshape(32, 32)
        #image = numpy.dstack((r, g, b))
        image = self.train_images[index]
        if self.transform:
            image = self.transform(np.uint8(image))
        if self.ages is not None:
            age = self.ages[index]
            return image, label, age
        else:
            return image, label
