import os
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.autograd import Variable
from copy import deepcopy
import time
from tqdm import tqdm
import numpy as np
import random
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.optim as optim

def psuedoImage_filtering(net,classify_net,loss_classify,previous_samples,previous_labels,samples_needed,seed=7,batch_size=128):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    to_pil = transforms.ToPILImage()
    loss_per_sample = [0.0 for i in range(len(previous_samples))]
    loss_per_sample = np.array(loss_per_sample)
    pseudo_images = []

    previous_samples = np.array(previous_samples)
    previous_labels = np.array(previous_labels)
    previous_samples = torch.from_numpy(previous_samples)
    previous_labels = torch.from_numpy(previous_labels)
    previous_samples = previous_samples.float()

    permutation = torch.randperm(previous_samples.size()[0])
    changed_indices = []
    net.eval()
    classify_net.eval()
    for i in range(0,previous_samples.size()[0],batch_size):
        indices = permutation[i:i+batch_size]
        samples = previous_samples[indices]
        labels = previous_labels[indices]
        samples = Variable(samples)
        labels = Variable(labels)
        samples = samples.cuda()
        labels = labels.cuda()
        _,rec_img,_ = net(input_data=None,embeddings=samples)
        #save_image(rec_img.data,'images/pseudo_class:{}_im_number:{}.png'.format(labels[0],i),nrow=16,normalize=True)
        #rec_img = to_pil(rec_img)
        output = classify_net(rec_img)
        loss_cl = loss_classify(output,labels)
        indices = indices.detach().numpy()
        loss_cl = loss_cl.cpu().detach().numpy()
        loss_per_sample[indices] = loss_cl
        rec_img = rec_img.cpu().detach().numpy()
        rec_img = rec_img.transpose(0,2,3,1)
        pseudo_images.extend(list(rec_img))
    pseudo_images = np.array(pseudo_images)
    sorted_indices = np.argsort(loss_per_sample)
    pseudo_images = pseudo_images[sorted_indices]
    pseudo_images = pseudo_images[0:samples_needed]
    pseudo_images = list(pseudo_images)

    for i in range(0,len(pseudo_images)):
        pseudo_images[i]-=pseudo_images[i].min()
        pseudo_images[i]*=255.0/pseudo_images[i].max()
        pseudo_images[i] = pseudo_images[i].astype(np.uint8)
    #pseudo_images = np.array(pseudo_images)
    return pseudo_images

def train_reconstruction(net,dataloaders_train_reconstruction,optimizer_rec,loss_rec,lambda_based = None,seed=7,classify_net=None,loss_classify=None,epoch=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    net.train()
    total_loss = []
    limiter = 20
    i = 0
    for images, labels in dataloaders_train_reconstruction:
        images = Variable(images)
        images = images.cuda()

        optimizer_rec.zero_grad()
        _,img_rec,_ = net(images)
        loss = (1-my_lambda) * loss_rec(img_rec,images)
        if classify_net is not None:
            labels = Variable(labels)
            labels = labels.cuda()
            classify_net.eval()
            _,output_original = classify_net(images,style_output=True)
            _,output_rec = classify_net(img_rec,style_output=True)
            loss_cl = loss_rec(output_rec,output_original)
            loss = loss + (my_lambda*loss_cl)
        loss.backward()
        total_loss.append(loss.item())
        optimizer_rec.step()
        i+=1
    return np.average(total_loss)

def eval_reconstruction(net,dataloaders_test_reconstruction,loss_rec,seed=7):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    net.eval()
    test_loss = 0.0
    for images, labels in dataloaders_test_reconstruction:
        images = Variable(images)

        images = images.cuda()

        _,outputs,_ = net(images)
        loss = loss_rec(outputs, images)
        test_loss += loss.item()
    print('Test set: Average loss: {:.4f}'.format(
        test_loss / len(dataloaders_test_reconstruction.dataset)
    ))
    return test_loss / len(dataloaders_test_reconstruction.dataset)

def get_embeddings(net,dataloaders_train_reconstruction,total_classes,increment=0,seed=7):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    embeddings = [[] for i in range(0,total_classes)]
    net.eval()
    j=0
    for images, labels in dataloaders_train_reconstruction:
        images = Variable(images)
        images = images.cuda()

        _,img_rec,emb = net(images)
        emb = emb.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        for i in range(len(labels)):
            embeddings[labels[i]-(increment*total_classes)].append(emb[i])
        j+=1
    return embeddings

def get_pseudoimages(net,full_embeddings,seed=7,class_number=0,global_counter=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    pseudo_images = []
    net.eval()
    limiter = 50
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    permutation = torch.randperm(full_embeddings.size()[0])

    # create a directory for saving the reconstructed images for this class
    current_directory = os.path.join('/home/ali/Ali_Work/clean_autoencoder_based/Imagenet-50/previous_classes',str(class_number))
    if os.path.exists(current_directory) == False:
        os.mkdir(current_directory)
    current_counter = global_counter
    for i in range(0,full_embeddings.size()[0],128):
        index = permutation[i:i+128]
        emb = full_embeddings[index]
        label = np.array([class_number for x in range(0,128)])
        label = torch.from_numpy(label)
        emb = Variable(emb)
        label = Variable(label)
        emb = emb.cuda()
        label = label.cuda()
        _,rec_img,_ = net(input_data=None,embeddings=emb)
        for j in range(0,len(rec_img)):
            if os.path.exists(os.path.join(current_directory,str(i+j+global_counter)+'.png')) == False:
                save_image(rec_img.cpu().data[j],os.path.join(current_directory,str(i+j+global_counter)+'.png'),normalize=True)
                current_counter+=1
        #save_image(rec_img.data,'images/class:{}_im_number:{}.png'.format(class_number,i),nrow=1,normalize=True)
        rec_img[:,0,:,:] = rec_img[:,0,:,:]*std[0] + mean[0]
        rec_img[:,1,:,:] = rec_img[:,1,:,:]*std[1] + mean[1]
        rec_img[:,2,:,:] = rec_img[:,2,:,:]*std[2] + mean[2]
        rec_img = rec_img.cpu()
        for j in range(0,len(rec_img)):
            pil_image = transforms.ToPILImage(mode='RGB')(rec_img[j])
            pil_image = np.asarray(pil_image)
            pseudo_images.append(pil_image)
    pseudo_images = np.array(pseudo_images)
    global_counter = current_counter
    return pseudo_images,global_counter

def train(classify_net,dataloaders_train_classification,optimizer,loss_classify,lambda_based=None,seed=7):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    classify_net.train()

    total_loss = []
    for images, labels in dataloaders_train_classification:
        images = Variable(images)
        labels = Variable(labels)
        images = images.cuda()
        optimizer.zero_grad()
        outputs = classify_net(images)

        images = None
        labels = labels.cuda()
        loss = my_lambda * loss_classify(outputs, labels)
        loss.backward()
        total_loss.append(loss.item())
        optimizer.step()
    return np.average(total_loss)

def train_with_decay(classify_net,dataloaders_train_classification,optimizer,loss_classify,seed=7):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    classify_net.train()
    total_loss = []
    for images, labels, ages in dataloaders_train_classification:
        images = Variable(images)
        labels = Variable(labels)
        ages = Variable(ages)

        labels = labels.cuda()
        images = images.cuda()
        ages = ages.cuda()

        optimizer.zero_grad()
        outputs = classify_net(images)
        loss = loss_classify(outputs, labels)
        loss = ages * loss
        loss = loss.mean()
        loss.backward()
        total_loss.append(loss.item())
        optimizer.step()
    return np.average(total_loss)

def eval_training(classify_net,dataloaders_test_classification,loss_classify,seed=7):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    classify_net.eval()
    test_loss = 0.0 # cost function error
    correct = 0.0
    for images, labels in dataloaders_test_classification:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = classify_net(images)
        images = None
        loss = loss_classify(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(dataloaders_test_classification.dataset),
        correct.float() / len(dataloaders_test_classification.dataset)
    ))
    return correct.float() / len(dataloaders_test_classification.dataset)

def eval_training_with_decay(classify_net,dataloaders_test_classification,loss_classify,seed=7):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    classify_net.eval()
    test_loss = 0.0
    correct = 0.0
    for images, labels in dataloaders_test_classification:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()
        outputs = classify_net(images)
        loss = loss_classify(outputs, labels)
        loss = loss.mean()
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(dataloaders_test_classification.dataset),
        correct.float() / len(dataloaders_test_classification.dataset)
    ))
    return correct.float() / len(dataloaders_test_classification.dataset)
