import numpy as np
import sys
import os
import time
import pickle
from PIL import Image
from copy import deepcopy
import random
from sklearn.model_selection import train_test_split
import json
#from multiprocessing import Pool as cpu_pool

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.resnet import resnet18
from models.resnet import resnet34
import torch.nn.functional as F

from get_incremental_data import getIncrementalData
from get_transformed_data_with_decay import getTransformedData
from get_previous_data import getPreviousData
#from get_transformed_data import getTransformedData
from my_models.new_shallow import auto_shallow
from training_functions import train_reconstruction
from training_functions import eval_reconstruction
from training_functions import get_embeddings
from training_functions import get_pseudoimages
from training_functions import train
from training_functions import eval_training
from training_functions import train_with_decay
from training_functions import eval_training_with_decay
from get_centroids import getCentroids
from Functions_new import get_pseudoSamples
from label_smoothing import LSR

seed=random.randint(0,10000)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

if __name__ == '__main__':
    dataset_name = 'imagenet'
    save_data = False

    use_saved_images = False
    path_to_previous = '/home/ali/Ali_Work/clean_autoencoder_based/Imagenet-50/previous_classes'
    validation_based = False

    if dataset_name == 'imagenet':
        path_to_train = '/media/ali/860 Evo/ali/ILSVRC2012_Train'
        path_to_test = '/media/ali/860 Evo/ali/ILSVRC2012_Test'

        # incremental steps info
        total_classes = 10
        full_classes = 1000
        limiter = 50

        # Image transformation mean and std
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

    # hyperparameters
    weight_decay = 5e-4
    classify_lr = 0.1
    reconstruction_lr = 0.001
    reconstruction_epochs = 100
    classification_epochs = 200
    batch_size = 128

    # for centroids
    get_covariances = True
    diag_covariances = True
    clustering_type = 'k_means'         # can use other clustering types from Functions_new
    centroids_limit = 25000
    centroid_finder = getCentroids(None,None,total_classes,seed=seed,get_covariances=get_covariances,diag_covariances=diag_covariances,centroids_limit=centroids_limit)

    features_name = 'multiple_'+str(centroids_limit)

    # autoencoders_set
    auto_1 = auto_shallow(total_classes,seed=seed)
    auto_2 = auto_shallow(total_classes,seed=seed)
    auto_3 = auto_shallow(total_classes,seed=seed)
    auto_4 = auto_shallow(total_classes,seed=seed)
    auto_5 = auto_shallow(total_classes,seed=seed)
    auto_1.cuda()
    auto_2.cuda()
    auto_3.cuda()
    auto_4.cuda()
    auto_5.cuda()
    autoencoder_set = [auto_1,auto_2,auto_3,auto_4,auto_5]

    #classify_net
    classify_net = resnet18(total_classes)

    # loss functions and optimizers
    #loss_classify = nn.CrossEntropyLoss()
    loss_classify = LSR()
    loss_rec = nn.MSELoss()

    # get incremental data
    incremental_data_creator = getIncrementalData(path_to_train,path_to_test,full_classes=full_classes,seed=seed)
    incremental_data_creator.incremental_data(total_classes=total_classes,limiter=limiter)

    # define transforms
    transforms_classification_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean,imagenet_std)
    ])
    transforms_classification_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean,imagenet_std)
    ])
    transforms_reconstruction = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean,imagenet_std)
    ])

    ################################# INCREMENTAL LEARNING PHASE ##################################
    complete_x_train = []
    complete_y_train = []
    complete_x_test = []
    complete_y_test = []
    complete_centroids = []
    complete_covariances = []
    complete_centroids_num = []
    complete_samples = []
    original_complete_centroids = []
    original_complete_covariances = []
    original_complete_centroids_num = []
    Accus = []
    full_classes = limiter
    for increment in range(0,int(full_classes/total_classes)):
        print ('This is increment number: ',increment)
        # get data for the current increment
        train_images_increment,train_labels_increment,test_images_increment,test_labels_increment = incremental_data_creator.incremental_data_per_increment(increment)
        if increment==0:
            previous_images = deepcopy(train_images_increment)
            previous_labels = deepcopy(train_labels_increment)
        else:
            """ Regeneration of pseudo_images """
            # first generate psuedo samples from centroids and covariances
            previous_images = []
            previous_labels = []
            pack = []

            for i in range(0,len(complete_centroids_num)):
                # for filtering
                #pack.append([complete_centroids[i],complete_covariances[i],[750 for x in range(0,len(complete_centroids_num[i]))],i,diag_covariances,total_classes,
                #increment,seed])
                pack.append([complete_centroids[i],complete_covariances[i],complete_centroids_num[i],i,diag_covariances,total_classes,
                increment,seed])
            previous_samples,_ = get_pseudoSamples(pack)
            print ('pseudo_samples generated. Creating pseudo_images.')
            print (' ')

            temp_loss = LSR(reduction='none')
            for i in range(0,len(complete_centroids)):

                #samples_needed = 500 - len(complete_samples[i])
                samples_needed = sum(complete_centroids_num[i])
                # for filtering
                #psuedo_images = psuedoImage_filtering(net,classify_net,temp_loss,previous_samples[i],[i for x in range(len(previous_samples[i]))],samples_needed,seed=seed)
                # for no filtering
                temp = np.array(previous_samples[i])
                temp = torch.from_numpy(temp)
                temp = temp.float()
                psuedo_images,counter = get_pseudoimages(autoencoder_set[increment-1],temp,class_number=i,seed=seed)
                previous_images.extend(psuedo_images)
                previous_labels.extend([i for x in range(samples_needed)])

                temp = np.array(complete_samples[i])
                temp = torch.from_numpy(temp)
                temp = temp.float()
                psuedo_images,_ = get_pseudoimages(autoencoder_set[increment-1],temp,class_number=i,seed=seed,global_counter=counter)
                previous_images.extend(list(psuedo_images))
                previous_labels.extend([i for x in range(len(psuedo_images))])

            print ('actual previous images',np.array(previous_images).shape)
            print ('previous labels',np.array(previous_labels).shape)
            #print ('previous ages',np.array(ages).shape)

            # append new images
            previous_images.extend(train_images_increment)
            previous_labels.extend(train_labels_increment)
        print ('train images',np.array(previous_images).shape)
        print ('train labels',np.array(previous_labels).shape)

        # complete x test update with new classes' test images
        complete_x_test.extend(test_images_increment)
        complete_y_test.extend(test_labels_increment)

        if validation_based:
            # Creating a validation split
            x_train,x_test,y_train,y_test = train_test_split(previous_images,previous_labels,test_size=0.2,stratify=previous_labels)
        else:
            # otherwise just rename variables
            x_train = previous_images
            y_train = previous_labels
            #x_test = complete_x_test
            #y_test = complete_y_test

        ############################## Classifier Training ######################################

        # get dataloaders
        train_dataset_classification = getTransformedData(x_train,y_train,transform=transforms_classification_train,seed=seed)
        test_dataset_classification = getTransformedData(complete_x_test,complete_y_test,transform=transforms_classification_test,seed=seed)

        dataloaders_train_classification = torch.utils.data.DataLoader(train_dataset_classification,batch_size = batch_size,
        shuffle=True, num_workers = 4)
        dataloaders_test_classification = torch.utils.data.DataLoader(test_dataset_classification,batch_size = batch_size,
        shuffle=False, num_workers = 4)

        if validation_based:
            val_dataset_classification = getTransformedData(x_test,y_test,transform=transforms_classification_test,seed=seed)
            dataloaders_val_classification = torch.utils.data.DataLoader(val_dataset_classification,batch_size = batch_size,
            shuffle=False, num_workers = 4)

        # update classifier's fc layer and optimizer
        classify_net.fc = nn.Linear(512,total_classes+(total_classes*increment))
        optimizer = optim.SGD(classify_net.parameters(),lr=classify_lr,weight_decay=weight_decay,momentum=0.9)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,160], gamma=0.2) #learning rate decay
        classify_net = classify_net.cuda()

        # for faster training times after the first increment
        if increment>0:
            classification_epochs = 45
            train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[37], gamma=0.1) #learning rate decay
        # load the classifier from file if it has already been trained on the classes of this increment
        classifier_path = './checkpoint/'+str(total_classes+(increment*total_classes))+"classes_"+dataset_name
        if os.path.exists(classifier_path):
            classify_net.load_state_dict(torch.load(classifier_path))
            epoch_acc = eval_training_with_decay(classify_net,dataloaders_test_classification,loss_classify,seed=seed)
            Accus.append(epoch_acc.cpu().numpy().tolist())
        else:
            since=time.time()
            best_acc = 0.0
            for epoch in range(0, classification_epochs):
                classification_loss = train(classify_net,dataloaders_train_classification,optimizer,loss_classify,lambda_based=None,seed=seed)
                print ('epoch:', epoch, '  classification loss:', classification_loss, '  learning rate:', optimizer.param_groups[0]['lr'])
                train_scheduler.step(epoch)
                if validation_based:
                    epoch_acc = eval_training_with_decay(classify_net,dataloaders_val_classification,loss_classify,seed=seed)
                    if epoch_acc>=best_acc:
                        best_acc = epoch_acc
                        best_model_wts = deepcopy(classify_net.state_dict())
                print (' ')
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            if validation_based:
                #print ('best_acc',best_acc)
                classify_net.load_state_dict(best_model_wts)
            epoch_acc = eval_training(classify_net,dataloaders_test_classification,loss_classify,seed=seed)
            print ('test_acc',epoch_acc)
            Accus.append(epoch_acc.cpu().numpy().tolist())
            Accus.append(epoch_acc.cpu().numpy().tolist())
            if validation_based:
                torch.save(best_model_wts, "./checkpoint/"+str(total_classes+(increment*total_classes))+"classes_"+dataset_name)
            else:
                torch.save(classify_net.state_dict(),"./checkpoint/"+str(total_classes+(increment*total_classes))+"classes_"+dataset_name)

        ############################## Autoencoder Training ######################################

        # get dataloaders
        train_dataset_reconstruction = getTransformedData(train_images_increment,train_labels_increment,
        transform=transforms_reconstruction,seed=seed)
        test_dataset_reconstruction = getTransformedData(test_images_increment,test_labels_increment,transform=transforms_reconstruction,seed=seed)

        dataloaders_train_reconstruction = torch.utils.data.DataLoader(train_dataset_reconstruction,batch_size = batch_size,
        shuffle=True, num_workers = 4)
        dataloaders_test_reconstruction = torch.utils.data.DataLoader(test_dataset_reconstruction,batch_size = batch_size,
        shuffle=True, num_workers = 4)
        for_embeddings_dataloader = torch.utils.data.DataLoader(train_dataset_reconstruction,batch_size = batch_size,
        shuffle=False, num_workers = 4)

        # no need to run autoencoder in the last increment
        if increment < 4:
            # path to load autoencoder from file if it has already been trained on the classes of this increment
            autoencoder_path = './checkpoint/autoencoder_'+str(total_classes+(increment*total_classes))+"classes_"+dataset_name

            if os.path.exists(autoencoder_path):
                autoencoder_set[increment].load_state_dict(torch.load(autoencoder_path))
            else:
                optimizer_rec = optim.Adam(autoencoder_set[increment].parameters(), lr=reconstruction_lr, weight_decay=weight_decay)
                train_scheduler_rec = optim.lr_scheduler.MultiStepLR(optimizer_rec, milestones=[50], gamma=0.1) #learning rate decay

                since=time.time()
                best_loss = 100.0
                for epoch in range(1, reconstruction_epochs):
                    #reconstruction_loss = train_reconstruction(autoencoder_set[increment],dataloaders_train_reconstruction,
                    #optimizer_rec,loss_rec,lambda_based=True,classify_net=classify_net,seed=seed,epoch=epoch)

                    reconstruction_loss = train_reconstruction(autoencoder_set[increment],dataloaders_train_reconstruction,optimizer_rec,loss_rec,seed=seed,epoch=epoch)
                    print ('epoch:', epoch, ' reconstruction loss:', reconstruction_loss)
                    train_scheduler_rec.step(epoch)

                    """
                    #test_loss = eval_reconstruction(net,dataloaders_test_reconstruction,loss_rec,seed=seed)
                    test_loss = eval_reconstruction(autoencoder_set[increment],dataloaders_test_reconstruction,loss_rec,seed=seed)
                    if test_loss<=best_loss:
                        best_loss = test_loss
                        #best_model_wts = deepcopy(net.state_dict())
                        best_model_wts = deepcopy(autoencoder_set[increment].state_dict())
                    """
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                print (' ')
                #autoencoder_set[increment].load_state_dict(best_model_wts)
                if validation_based:
                    torch.save(best_model_wts, "./checkpoint/autoencoder_"+str(total_classes+(increment*total_classes))+"classes_"+dataset_name)
                else:
                    torch.save(autoencoder_set[increment].state_dict(),
                    "./checkpoint/autoencoder_"+str(total_classes+(increment*total_classes))+"classes_"+dataset_name)

            # get embeddings from the trained autoencoder
            embeddings = get_embeddings(autoencoder_set[increment],for_embeddings_dataloader,total_classes,seed=seed,increment=increment)
            print ('embeddings',np.array(embeddings).shape)

            distance_threshold = int(centroids_limit/((increment*total_classes)+total_classes))

            # 1300*10 = 13000 => save all samples, no need to perform clustering
            if distance_threshold>=1300:
                temp = list(embeddings)
                complete_samples.extend(temp)
                original_complete_centroids.extend(temp)
                complete_centroids = [[] for x in range(total_classes+(total_classes*increment))]
                complete_centroids_num = [[] for x in range(total_classes+(total_classes*increment))]
                complete_covariances = [[] for x in range(total_classes+(total_classes*increment))]
                for i in range(0,len(temp)):
                    original_complete_centroids_num.append([1 for x in range(0,len(temp[i]))])
                    original_complete_covariances.append([[1.0 for x in range(0,len(temp[i][0]))] for y in range(0,len(temp[i]))])
            else:
                # initialize the centroid finder variable
                centroid_finder.initialize(None,None,total_classes,increment=0,d_base=distance_threshold,get_covariances=get_covariances,
                diag_covariances=diag_covariances,seed=seed,current_centroids=original_complete_centroids,
                complete_covariances=original_complete_covariances,complete_centroids_num=original_complete_centroids_num,clustering_type=clustering_type,
                centroids_limit=centroids_limit)

                # find clusters
                centroid_finder.without_validation(embeddings)
                complete_centroids = centroid_finder.complete_centroids
                complete_covariances = centroid_finder.complete_covariances
                complete_centroids_num = centroid_finder.complete_centroids_num
                original_complete_centroids = deepcopy(complete_centroids)
                original_complete_covariances = deepcopy(complete_covariances)
                original_complete_centroids_num = deepcopy(complete_centroids_num)

                complete_samples = [[] for x in range(0,len(complete_centroids))]
                # separate samples and centroids from the output by clustering
                cur_num_cent = 0
                for i in range(0,len(complete_centroids)):
                    sample_indices = [j for j,x in enumerate(complete_centroids_num[i]) if x==1]
                    temp = np.array(complete_centroids[i])
                    complete_samples[i] = temp[sample_indices]
                    sample_indices = [j for j,x in enumerate(complete_centroids_num[i]) if x!=1]
                    temp = np.array(complete_centroids[i])
                    complete_centroids[i] = temp[sample_indices]
                    temp = np.array(complete_centroids_num[i])
                    complete_centroids_num[i] = temp[sample_indices]
                    temp = np.array(complete_covariances[i])
                    complete_covariances[i] = temp[sample_indices]
                    print ('for class',i,'samples are',np.array(complete_samples[i]).shape)
                    print ('for class',i,'centroids are',np.array(complete_centroids[i]).shape)
                    cur_num_cent += len(complete_centroids[i])

    print ('All accuracies yet', Accus)
    experimental_data = dict()
    experimental_data['seed'] = seed
    experimental_data['acc'] = Accus
    experimental_data['centroids_limit'] = centroids_limit
    experimental_data['current_centroids'] = cur_num_cent
    if save_data == True:
        with open('data.json','r') as f:
            data=json.load(f)
        if features_name not in data:
            data[features_name] = dict()
        data[features_name][str(len(data[features_name])+1)] = experimental_data
        with open('data.json', 'w') as fp:
            json.dump(data, fp, indent=4, sort_keys=True)
