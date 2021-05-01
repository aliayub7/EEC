"""
Created on 03/11/2020

@author: Ali Ayub
"""

import numpy as np
import random
from sklearn.cluster import KMeans
from scipy.spatial import distance
from copy import deepcopy
import math
from multiprocessing import Pool
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fclusterdata
from scipy.cluster.hierarchy import fcluster, ward, average, weighted, complete, single
from scipy.spatial.distance import pdist
import os
os.environ["OMP_NUM_THREADS"] = "1"

distance_metric = 'euclidean'
def get_centroids(train_pack):
    # unpack x_train
    x_train = train_pack[0]
    distance_threshold = train_pack[1]
    clustering_type = train_pack[2]
    get_covariances = train_pack[3]
    diag_covariances = train_pack[4]

    if clustering_type == 'Agglomerative':
        dist_mat=pdist(x_train,metric='euclidean')
        Z = weighted(dist_mat)
        dn = hierarchy.dendrogram(Z)
        labels=fcluster(Z, t=distance_threshold, criterion='distance')
        centroids = [[] for y in range(0,max(labels))]
        total_num = [0 for x in range(0,max(labels))]
        per_labels = [[] for y in range(0,max(labels))]
        for j in range(len(x_train)):
            per_labels[labels[j]-1].append(x_train[j])
            total_num[labels[j]-1]+=1
        covariances = [[] for y in range(0,max(labels))]

        for j in range(0,max(labels)):
            centroids[j] = np.mean(per_labels[j],0)
            if get_covariances==True:
                if diag_covariances != True:
                    covariances[j] = np.cov(np.array(per_labels[j]).T)
                else:
                    temp = np.cov(np.array(per_labels[j]).T)
                    covariances[j] = temp.diagonal()
        #for j in range(0,len(x_train)):
        #    centroids[labels[j]-1]+=x_train[j]
        #    total_number[labels[j]-1]+=1
        #for j in range(0,len(centroids)):
        #    centroids[j] = np.divide(centroids[j],total_number[j])

    elif clustering_type == 'Agglomerative_variant':
        # for each training sample do the same stuff...
        if len(x_train)>0:
            centroids = [[0 for x in range(len(x_train[0]))]]
            for_cov = [[]]
            labels = []
            # initalize centroids
            centroids[0] = x_train[0]
            for_cov[0].append(x_train[0])
            total_num = [1]
            labels.append(0)
            for i in range(1,len(x_train)):
                distances=[]
                indices = []
                for j in range(0,len(centroids)):
                    d = find_distance(x_train[i],centroids[j],distance_metric)
                    if d<distance_threshold:
                        distances.append(d)
                        indices.append(j)
                if len(distances)==0:
                    centroids.append(x_train[i])
                    total_num.append(1)
                    for_cov.append([])
                    for_cov[len(for_cov)-1].append(list(x_train[i]))
                    labels.append(len(centroids)-1)
                else:
                    #min_d = np.argmin(distances)
                    #centroids[indices[min_d]] = np.add(centroids[indices[min_d]],x_train[i])
                    #total_num[indices[min_d]]+=1
                    min_d = np.argmin(distances)
                    centroids[indices[min_d]] = np.add(np.multiply(total_num[indices[min_d]],centroids[indices[min_d]]),x_train[i])
                    total_num[indices[min_d]]+=1
                    centroids[indices[min_d]] = np.divide(centroids[indices[min_d]],(total_num[indices[min_d]]))
                    for_cov[indices[min_d]].append(list(x_train[i]))
                    labels.append(indices[min_d])
            # calculate covariances
            if get_covariances==True:
                covariances = deepcopy(for_cov)
                for j in range(0,len(for_cov)):
                    if total_num[j]>1:
                        if diag_covariances != True:
                            covariances[j] = np.cov(np.array(for_cov[j]).T)
                        else:
                            temp = np.cov(np.array(for_cov[j]).T)
                            covariances[j] = temp.diagonal()
                    else:
                        covariances[j] = np.array([1.0 for x in range(0,len(x_train[0]))])

                #or j in range(0,len(total_num)):
                #    centroids[j]=np.divide(centroids[j],total_num[j])
        else:
            centroids = []

    elif clustering_type == 'k_means':
        kmeans = KMeans(n_clusters=distance_threshold, random_state = 0).fit(x_train)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        total_num = [0 for x in range(0,max(labels)+1)]
        per_labels = [[] for y in range(0,max(labels)+1)]
        for j in range(len(x_train)):
            per_labels[labels[j]].append(x_train[j])
            total_num[labels[j]]+=1
        covariances = [[] for y in range(0,max(labels)+1)]
        for j in range(0,max(labels)+1):
            if get_covariances==True:
                if total_num[j]>1:
                    if diag_covariances != True:
                        covariances[j] = np.cov(np.array(per_labels[j]).T)
                    else:
                        temp = np.cov(np.array(per_labels[j]).T)
                        covariances[j] = temp.diagonal()
                else:
                    covariances[j] = np.array([1.0 for x in range(0,len(x_train[0]))])
    elif clustering_type == 'NCM':
        centroids = [[0 for x in range(len(x_train[0]))]]
        centroids[0] = np.average(x_train,0)

    if get_covariances == True:
        return [centroids,covariances,total_num,labels]
    else:
        return centroids

def find_distance(data_vec,centroid,distance_metric):
    if distance_metric=='euclidean':
        return np.linalg.norm(data_vec-centroid)
    elif distance_metric == 'euclidean_squared':
        return np.square(np.linalg.norm(data_vec-centroid))
    elif distance_metric == 'cosine':
        return distance.cosine(data_vec,centroid)

# reduce give centroids using k-means
def reduce_centroids(centroid_pack):
    centroids = centroid_pack[0]
    reduction_per_class = centroid_pack[1]
    n_clusters = len(centroids) - reduction_per_class
    if n_clusters>0:

        #out_centroids = get_centroids([centroids,n_clusters,'k_means',True,False])
        kmeans = KMeans(n_clusters=n_clusters, random_state = 0).fit(centroids)
        out_centroids = kmeans.cluster_centers_

        # simple reduction
        #out_centroids = deepcopy(centroids)
        #del out_centroids[0:reduction_per_class]
    else:
        out_centroids = centroids
    return out_centroids

# check if the centroids should be reduced and reduce them
def check_reduce_centroids(temp_complete_centroids,current_total_centroids,temp_exp_centroids,total_centroids_limit,increment,total_classes):

    if current_total_centroids + temp_exp_centroids > total_centroids_limit:
        reduction_centroids = current_total_centroids + temp_exp_centroids - total_centroids_limit
        classes_so_far = increment*total_classes
        centroid_pack = []
        for i in range(0,len(temp_complete_centroids)):
            reduction_per_class = round((len(temp_complete_centroids[i])/current_total_centroids)*reduction_centroids)
            centroid_pack.append([temp_complete_centroids[i],reduction_per_class])
        my_pool = Pool(len(temp_complete_centroids))
        temp_complete_centroids = my_pool.map(reduce_centroids,centroid_pack)
        my_pool.close()
    return temp_complete_centroids

# reduce given centroids and covariances using k-means. CURRENTLY ONLY FOR DIAGONAL COVARIANCES
def reduce_centroids_covariances(centroid_pack):
    centroids = centroid_pack[0]
    covariances = centroid_pack[1]
    centroid_num = centroid_pack[2]
    reduction_per_class = centroid_pack[3]
    n_clusters = len(centroids) - reduction_per_class

    if n_clusters>0:

        #out_centroids = get_centroids([centroids,n_clusters,'k_means',False])
        kmeans = KMeans(n_clusters=n_clusters, random_state = 0).fit(centroids)
        out_centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        out_covariances = []
        covariances = np.array(covariances)
        centroid_num = np.array(centroid_num)
        out_centroids_num = []
        #print ('these are the labels',labels)
        for i in range(0,n_clusters):
            indices = [x for x,j in enumerate(labels) if j==i]
            temp = covariances[indices]
            temp_num = centroid_num[indices]
            #print ('before tem shape',temp.shape)
            #print ('this is temp',temp)
            #print ('this is temp_num',temp_num)
            #print ('this is covariances',covariances)
            #input('continue')
            if 1 in temp_num:
                if(len(set(temp_num))==1):
                    #print (temp_num)
                    out_covariances.append(temp[0])
                else:
                    temp = np.delete(temp,np.argwhere(temp_num==1),0)
                    #print ('temp',temp.shape)
                    out_covariances.append(np.mean(temp,0))
                    #print ('this is cov shape',np.array(out_covariances).shape)
                    #print ('this is cov',out_covariances)
                    #input ('continue')
            else:
                out_covariances.append(np.mean(temp,0))
            out_centroids_num.append(np.sum(temp_num,0))

        #out_centroids = deepcopy(centroids)
        #out_covariances = deepcopy(covariances)
        #out_centroids_num = deepcopy(centroid_num)
        #del out_centroids[0:reduction_per_class]
        #del out_covariances[0:reduction_per_class]
        #del out_centroids_num[0:reduction_per_class]
    else:
        out_centroids = centroids
        out_covariances = covariances
        out_centroids_num = centroid_num

    # simple reduction

    return out_centroids,out_covariances,out_centroids_num

# check if the centroids should be reduced and reduce them
def check_reduce_centroids_covariances(temp_complete_centroids,temp_complete_covariances,temp_complete_centroid_num,
    current_total_centroids,temp_exp_centroids,total_centroids_limit,increment,total_classes):
    if current_total_centroids + temp_exp_centroids > total_centroids_limit:
        reduction_centroids = current_total_centroids + temp_exp_centroids - total_centroids_limit
        classes_so_far = increment*total_classes
        centroid_pack = []
        for i in range(0,len(temp_complete_centroids)):
            reduction_per_class = round((len(temp_complete_centroids[i])/current_total_centroids)*reduction_centroids)
            centroid_pack.append([temp_complete_centroids[i],temp_complete_covariances[i],temp_complete_centroid_num[i],reduction_per_class])
        my_pool = Pool(len(temp_complete_centroids))
        outer = my_pool.map(reduce_centroids_covariances,centroid_pack)
        my_pool.close()

        #outer = []
        #for i in range(0,len(centroid_pack)):
        #    outer.append(reduce_centroids_covariances(centroid_pack[i]))
        #input ('continue')
        temp_complete_centroids = []
        temp_complete_covariances = []
        temp_complete_centroid_num = []
        for i in range(0,len(outer)):
            temp_complete_centroids.append(outer[i][0])
            temp_complete_covariances.append(outer[i][1])
            temp_complete_centroid_num.append(outer[i][2])
    return temp_complete_centroids,temp_complete_covariances,temp_complete_centroid_num

def predict_multiple_class(pack):
    data_vec = pack[0]
    centroids = pack[1]
    class_centroid = pack[2]
    distance_metric = pack[3]
    if distance_metric=='euclidean':
        dists = np.subtract(data_vec,centroids)
        dists = np.linalg.norm(dists,axis=1)
    dist = [[dists[x],class_centroid] for x in range(len(centroids))]
    return dist
def predict_multiple_k(data_vec,centroids,distance_metric,tops,weighting):
    dist = []
    for i in range(0,len(centroids)):
        temp = predict_multiple_class([data_vec,centroids[i],i,distance_metric])
        dist.extend(temp)
    sorted_dist = sorted(dist)
    common_classes = [0]*len(centroids)
    # for all k values
    all_tops = []
    for k in range(0,tops+1):
        if k<len(sorted_dist):
            if sorted_dist[k][0]==0.0:
                common_classes[sorted_dist[k][1]] += 1
            else:
                common_classes[sorted_dist[k][1]] += ((1/(k+1))*
                                                    ((sorted_dist[len(sorted_dist)-1][0]-sorted_dist[k][0])/(sorted_dist[len(sorted_dist)-1][0]-sorted_dist[0][0])))
            common_classes = np.divide(common_classes,sum(common_classes))
            common_classes = np.multiply(common_classes,weighting)
            all_tops.append(np.argmax(common_classes))
        else:
            all_tops.append(-1)
    return all_tops

def get_accu (pack):
    x_test = pack[0]
    y_test = pack[1]
    centroids = pack[2]
    distance_metric = pack[3]
    k = pack[4]
    weighting = pack[5]
    increment = pack[6]
    batch_size = pack[7]

    accus = [[0.0 for x in range(batch_size)] for y in range(k)]
    total_labels = [0.0 for x in range(batch_size)]
    for i in range(0,len(y_test)):
        total_labels[y_test[i]-(increment*batch_size)]+=1
        predicted_label=predict_multiple_k(x_test[i],centroids,distance_metric,k,weighting)
        for j in range(0,k):
            accus[j][y_test[i]-(increment*batch_size)]+=(predicted_label[j]==y_test[i])
    return [accus,total_labels]

# get validation accuracy for all the k values
def get_validation_accuracy(test_pack):
    x_test = test_pack[0]
    y_test = test_pack[1]
    centroids = test_pack[2]
    k = test_pack[3]
    batch_size = test_pack[4]
    increment = test_pack[5]
    weighting = test_pack[6]

    accus = [[0.0 for x in range(batch_size)] for y in range(k)]
    total_labels = [0.0 for x in range(batch_size)]
    acc=0

    # divide y_test in 24 equal segments
    how_many = round(len(y_test)/24)
    pack = []
    now=0
    while now<len(y_test):
        if now+how_many>=len(y_test):
            pack.append([x_test[now:len(y_test)],y_test[now:len(y_test)],centroids,'euclidean',k,weighting,increment,batch_size])
        else:
            pack.append([x_test[now:how_many+now],y_test[now:how_many+now],centroids,'euclidean',k,weighting,increment,batch_size])
        now+=how_many

    my_pool = Pool(25)
    return_pack = my_pool.map(get_accu,pack)
    my_pool.close()
    for i in range(0,len(return_pack)):
        accus = np.sum([accus,return_pack[i][0]],axis=0)
        total_labels = np.sum([total_labels,return_pack[i][1]],axis=0)

    for i in range(0,batch_size):
        if total_labels[i]>0:
            for j in range(0,k):
                accus[j][i] = accus[j][i]/total_labels[i]
        else:
            for j in range(0,k):
                accus[j][i]=1.0

    #acc = np.mean(accus)
    acc = [np.mean(accus[j]) for j in range(k)]
    return acc

# get validation accuracy for all the k values
def get_validation_accuracy_fearnet(test_pack):
    x_test = test_pack[0]
    y_test = test_pack[1]
    centroids = test_pack[2]
    k = test_pack[3]
    batch_size = test_pack[4]
    increment = test_pack[5]
    weighting = test_pack[6]
    base_classes = test_pack[7]

    accus = [[0 for x in range(batch_size)] for y in range(k)]
    total_labels = [0 for x in range(batch_size)]
    acc=0
    for i in range(0,len(y_test)):
        total_labels[y_test[i]-(increment*batch_size)-base_classes]+=1
        predicted_label=predict_multiple_k(x_test[i],centroids,'euclidean',k,weighting)
        for j in range(0,k):
            accus[j][y_test[i]-(increment*batch_size)-base_classes]+=(predicted_label[j]==y_test[i])

    for i in range(0,batch_size):
        if total_labels[i]>0:
            for j in range(0,k):
                accus[j][i] = accus[j][i]/total_labels[i]
        else:
            for j in range(0,k):
                accus[j][i]=1.0
    acc = [np.mean(accus[j]) for j in range(k)]
    return acc

def get_pseudoSamples_hidden(pack):
    complete_centroids = pack[0]
    complete_covariances = pack[1]
    complete_centroids_num = pack[2]
    label = pack[3]
    diag_covariances = pack[4]
    seed = pack[7]
    #seed = pack[10]
    np.random.seed(seed)
    random.seed(seed)
    previous_samples = []
    previous_labels = []
    for j in range(0,len(complete_centroids_num)):
        if complete_centroids_num[j]>1:
            if diag_covariances != True:
                temp = list(np.random.multivariate_normal(complete_centroids[j],complete_covariances[j],complete_centroids_num[j]))
            else:
                temp = list(np.random.multivariate_normal(complete_centroids[j],np.diag(complete_covariances[j]),complete_centroids_num[j]))
            previous_samples.extend(temp)
            previous_labels.extend([label for x in range(0,complete_centroids_num[j])])
        else:
            previous_samples.append(complete_centroids[j])
            previous_labels.append(label)
    return [previous_samples,previous_labels]

def get_pseudoSamples(pack):
    total_classes = pack[0][5]
    increment = pack[0][6]
    seed = pack[0][7]
    np.random.seed(seed)
    random.seed(seed)
    my_pool = Pool(total_classes+(increment*total_classes))
    return_pack = my_pool.map(get_pseudoSamples_hidden,pack)
    my_pool.close()
    previous_samples = []
    previous_labels = []
    for i in range(0,len(return_pack)):
        #previous_samples.extend(return_pack[i][0])
        #previous_labels.extend(return_pack[i][1])
        previous_samples.append(return_pack[i][0])
        previous_labels.append(return_pack[i][1])
    return previous_samples,previous_labels

# Currently only works with one class per increment because centroids for all classes are separate
def get_pseudoSamplesAges(pack):
    total_classes = pack[0][5]
    increment = pack[0][6]
    sample_decay_coeff = pack[0][7]
    decay_type = pack[0][8]
    separtate = pack[0][9]
    seed = pack[0][10]
    random.seed(seed)
    np.random.seed(seed)
    my_pool = Pool(total_classes+(increment*total_classes))
    return_pack = my_pool.map(get_pseudoSamples_hidden,pack)
    my_pool.close()

    if total_classes!=1:
        total_increments = int((len(return_pack))/total_classes)
    else:
        total_increments = int((len(return_pack))/total_classes) - 1
    previous_samples = []
    previous_labels = []
    ages = []
    for i in range(0,len(return_pack)):
        if total_classes==1 and i ==0:
            cur = 1
        else:
            cur = i
        #temp = int((len(return_pack) -1 - cur)/total_classes) + 1
        #temp = sum([1/x for x in range(1,temp+1)])
        temp = 1
        if separtate != True:
            previous_samples.extend(return_pack[i][0])
            previous_labels.extend(return_pack[i][1])
            if decay_type == 'exponential':
                ages.extend([np.exp(-((temp)*sample_decay_coeff)) for x in range(0,len(return_pack[i][1]))])
        else:
            previous_samples.append(return_pack[i][0])
            previous_labels.append(return_pack[i][1])
            if decay_type == 'exponential':
                ages.append([np.exp(-((temp)*sample_decay_coeff)) for x in range(0,len(return_pack[i][1]))])
    return previous_samples,previous_labels,ages
