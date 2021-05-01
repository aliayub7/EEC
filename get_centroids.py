# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 2020

@author: Ali Ayub
"""

import numpy as np
from copy import deepcopy
import math
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from Functions_new import get_centroids
from Functions_new import check_reduce_centroids_covariances
from Functions_new import check_reduce_centroids
from Functions_new import get_validation_accuracy
from sklearn.model_selection import KFold
import random
# THE FOLLOWING IS DEFINITELY NEEDED WHEN WORKING WITH PYTORCH
import os
os.environ["OMP_NUM_THREADS"] = "1"


class getCentroids:
    def __init__(self,x_train,y_train,classes,seed,centroids_limit=None,current_centroids=[],increment=None,distance_metric='euclidean',
    clustering_type='Agglomerative_variant',k_base=1,k_limit=25,x_val=None,y_val=None,get_covariances=False,complete_covariances=[],
    complete_centroids_num=[],d_base=17.0,d_limit=23.0,d_step=0.2,diag_covariances=False):
        self.x_train = x_train
        self.y_train = y_train
        self.total_classes = classes
        self.increment = increment
        self.total_centroids_limit = centroids_limit
        self.complete_centroids = current_centroids
        self.distance_metric = distance_metric
        self.clustering_type = clustering_type
        self.k_base = k_base
        self.k_limit = k_limit
        self.best_k = None
        self.total_num = []
        self.x_val = x_val
        self.y_val = y_val
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.get_covariances = get_covariances
        self.diag_covariances = diag_covariances
        self.complete_covariances = complete_covariances
        self.complete_centroids_num = complete_centroids_num
        self.d_base = d_base
        self.d_limit = d_limit
        self.d_step = d_step

    def initialize(self,x_train,y_train,classes,seed,centroids_limit=None,current_centroids=[],increment=None,distance_metric='euclidean',
    clustering_type='Agglomerative_variant',k_base=1,k_limit=25,x_val=None,y_val=None,get_covariances=False,complete_covariances=[],
    complete_centroids_num=[],d_base=17.0,d_limit=23.0,d_step=0.2,diag_covariances=False,original_image_indices = []):
        self.x_train = x_train
        self.y_train = y_train
        self.total_classes = classes
        self.increment = increment
        self.total_centroids_limit = centroids_limit
        self.complete_centroids = current_centroids
        self.distance_metric = distance_metric
        self.clustering_type = clustering_type
        self.k_base = k_base
        self.k_limit = k_limit
        self.best_k = None
        self.x_val = x_val
        self.y_val = y_val
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.get_covariances = get_covariances
        self.diag_covariances = diag_covariances
        self.complete_covariances = complete_covariances
        self.complete_centroids_num = complete_centroids_num
        self.original_image_indices = original_image_indices
        self.d_base = d_base
        self.d_limit = d_limit
        self.d_step = d_step

    def without_validation(self,train_data):
        current_total_centroids = 0
        for i in range(0,len(self.complete_centroids)):
            current_total_centroids+=len(self.complete_centroids[i])
        train_pack = []
        for i in range(0,self.total_classes):
            if self.d_base > len(train_data[i]):
                temp = len(train_data[i])
            else:
                temp = self.d_base
            train_pack.append([train_data[i],temp,self.clustering_type,self.get_covariances,self.diag_covariances])
            #train_pack.append([train_data[i],self.d_base,self.clustering_type,self.get_covariances,self.diag_covariances])
        if self.get_covariances!=True:
            my_pool = Pool(self.total_classes)
            centroids = my_pool.map(get_centroids,train_pack)
            my_pool.close()
        else:
            my_pool = Pool(self.total_classes)
            centroids_variances = my_pool.map(get_centroids,train_pack)
            my_pool.close()
            centroids = []
            covariances = []
            centroids_num = []
            image_indices = []
            for j in range(0,len(centroids_variances)):
                centroids.append(centroids_variances[j][0])
                covariances.append(centroids_variances[j][1])
                centroids_num.append(centroids_variances[j][2])
                #temp = np.add((j*len(train_data[0])),centroids_variances[j][3])
                #sorted_images_indices.extend(list(temp))
                temp = []
                tot = 0
                for k in range(0,len(centroids_variances[j][0])):
                    temp.append([x for x,y in enumerate(centroids_variances[j][3]) if y==k])
                    tot += len([x for x,y in enumerate(centroids_variances[j][3]) if y==k])
                image_indices.append(temp)
        exp_centroids = 0
        for i in range(0,len(centroids)):
            exp_centroids+=len(centroids[i])

        # reduce previous centroids if more than allowed UPDATE THIS FOR IMAGE INDICES
        if self.total_centroids_limit!=None:
            self.complete_centroids,self.complete_covariances,self.complete_centroids_num = check_reduce_centroids_covariances(self.complete_centroids,
            self.complete_covariances,self.complete_centroids_num,
            current_total_centroids,exp_centroids,self.total_centroids_limit,self.increment,self.total_classes)
        # add the new centroids to the complete_centroids
        self.complete_centroids.extend(centroids)
        if self.get_covariances==True:
            self.complete_covariances.extend(covariances)
            self.complete_centroids_num.extend(centroids_num)
        return image_indices
