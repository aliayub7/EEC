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

class getIncrementalData:
    def __init__(self,path_to_train,path_to_test,full_classes,seed):
        self.path_to_train = path_to_train
        self.path_to_test = path_to_test
        self.orig_lab = [i for i in range(0,full_classes)]
        self.full_classes = full_classes
        self.total_classes = 1
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def initialize(self,path_to_train,path_to_test,full_classes,seed):
        self.path_to_train = path_to_train
        self.path_to_test = path_to_test
        self.orig_lab = [i for i in range(0,full_classes)]
        self.full_classes = full_classes
        self.total_classes = 1
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def incremental_data(self,total_classes=10,limiter=None):
        self.orig_lab = [i for i in range(0,self.full_classes)]
        random.shuffle(self.orig_lab)
        self.total_classes = total_classes
        self.orig_lab = self.orig_lab[0:limiter]

    def incremental_data_per_increment(self,increment=0):
        self.orig_lab[1] = 0
        if increment == 0:
            classes = self.orig_lab[0:self.total_classes]
        else:
            classes = self.orig_lab[self.total_classes+((increment-1)*self.total_classes):self.total_classes+(increment*self.total_classes)]
        print ('classes to be loaded',classes)
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []

        pack = []
        for i in range(0,len(classes)):
            pack.append(os.path.join(self.path_to_train,str(classes[i])))
        my_pool = Pool(self.total_classes)
        return_pack = my_pool.map(get_images,pack)
        my_pool.close()
        for i in range(0,len(classes)):
            train_images.extend(return_pack[i])
            train_labels.extend([i+(increment*self.total_classes) for x in range(0,len(return_pack[i]))])

        pack = []
        for i in range(0,len(classes)):
            pack.append(os.path.join(self.path_to_test,str(classes[i])))
        my_pool = Pool(self.total_classes)
        return_pack = my_pool.map(get_images,pack)
        my_pool.close()
        for i in range(0,len(classes)):
            test_images.extend(return_pack[i])
            test_labels.extend([i+(increment*self.total_classes) for x in range(0,len(return_pack[i]))])
        return train_images,train_labels,test_images,test_labels
