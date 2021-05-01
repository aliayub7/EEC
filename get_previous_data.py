# -*- coding: utf-8 -*-
"""
Created on Wed Jul 3 2019

@author: Ali Ayub
"""
import numpy as np
from copy import deepcopy
import pickle
import math
import random
from multiprocessing import Pool
import os
from PIL import Image
os.environ["OMP_NUM_THREADS"] = "1"

def get_images(path):
    resolution = 32
    train_images = []
    for file in os.listdir(path):
        file_path = os.path.join(path,file)
        img = Image.open(file_path)
        #img = img.resize((32,32))
        img = img.resize((resolution,resolution))
        np_img = np.asarray(img)
        if len(np_img.shape)==3:
            train_images.append(np.asarray(img))
    return train_images

class getPreviousData:
    def __init__(self,path_to_train,total_classes,seed):
        self.path_to_train = path_to_train
        self.total_classes = total_classes
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def initialize(self,path_to_train,total_classes,seed):
        self.path_to_train = path_to_train
        self.orig_lab = [i for i in range(0,full_classes)]
        self.total_classes = total_classes
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def previous_data(self):
        classes = [i for i in range(self.total_classes)]
        print ('classes to be loaded',classes)
        train_images = []
        train_labels = []

        pack = []
        for i in range(0,len(classes)):
            pack.append(os.path.join(self.path_to_train,str(classes[i])))
        my_pool = Pool(self.total_classes)
        return_pack = my_pool.map(get_images,pack)
        my_pool.close()
        for i in range(0,len(classes)):
            train_images.extend(return_pack[i])
            train_labels.extend([i for x in range(0,len(return_pack[i]))])

        return train_images,train_labels
