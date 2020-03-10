import argparse
import os
import time
import copy
import six
import sys
import torch
import numpy as np

import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
### tensorboard

from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

writer = SummaryWriter('runs/experiment_prior')
#######
sys.path.append('/home/jbonato/Documents/cvpr_clvision_challenge/')
from core50.dataset import CORE50
from utils.train_test import train_net, test_multitask, preprocess_imgs
from utils.common import create_code_snapshot,check_ext_mem, check_ram_usage

######
import torch.nn as nn
import torch
import numpy

from DIM_model import *
from train import *
from PP_misc.loader import LoadDataset,data_split,data_split_Tr_CV,LoadFeatures
from PP_misc.transf import * 
from model import classifier

from train_prior_disc import save_prior_dist


device = torch.device('cuda:0')
dataset = CORE50(root='/home/jbonato/Documents/cvpr_clvision_challenge/core50/data/', scenario='multi-task-nc',preload=True)
test =  dataset.get_full_valid_set(reduced=False)


# data_test = data_testf[labels_testf<5,:,:,:]
# labels_test = labels_testf[labels_testf<5]

def convert_lab(labels,task,map_lb):
    if task!=0:
        case = map_lb[str(task)]
        print(case)
        if case == 'A':
            n_labels = labels%5
        if case =='B':
            n_labels = labels%5 + 5
        return n_labels
    else:
        return labels
def revert_lab(labels,task,map_lb):
    if task!=0:
        case = map_lb[str(task)]
        if case == 'A':
            n_labels = labels+task*5+5
        if case == 'B':
            n_labels = labels+task*5
        return n_labels
    else:
        return labels
case = {'0':None,'1':'B','2':'A','3':'B','4':'A','5':'A','6':'A','7':'B','8':'B'}
load = False
replay = True
train = True

stats = {"ram": [], "disk": []}

if train:
    tr = Transform(affine=0.5, train=True,cutout_ratio=0.6,ssr_ratio=0.6,flip = 0.6)
    for i, train_batch in enumerate(dataset):
        writerDIM = SummaryWriter('runs/experiment_DIM'+str(i))
        data,labelsI, t = train_batch
        labels = convert_lab(labelsI,t,case)
        index_tr,index_cv,coreset = data_split(data.shape[0],777)
        if i == 0:
            ext_mem = [data[coreset], labels[coreset]]
            dataC = np.concatenate((data[index_tr], data[index_cv]),axis=0)
            labC = np.concatenate((labels[index_tr],labels[index_cv]),axis=0)
        else:
            dataP = ext_mem[0]
            labP = ext_mem[1]

            ext_mem = [
                np.concatenate((data[coreset], ext_mem[0])),
                np.concatenate((labels[coreset], ext_mem[1]))]
            if replay:
                dataC = np.concatenate((data[index_tr], data[index_cv],dataP),axis=0)
                labC = np.concatenate((labels[index_tr],labels[index_cv],labP),axis=0)
            else:
                dataC = np.concatenate((data[index_tr], data[index_cv]),axis=0)
                labC = np.concatenate((labels[index_tr],labels[index_cv]),axis=0)
        
        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())
        
        print("----------- batch {0} -------------".format(i))
        print("Task Label: ", t)
        trC,cvC = data_split_Tr_CV(dataC.shape[0],777)

        train_set = LoadDataset(dataC,labC,transform=tr,indices=trC)
        val_set = LoadDataset(dataC,labC,transform=tr,indices=cvC)
        print('Training set: {0} \n Validation Set {1}'.format(train_set.__len__(),val_set.__len__()))
        batch_size=64
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        dataloaders = {'train':train_loader,'val':valid_loader}

        if i ==0:        
            prior = False
            ep=50
            dim_model = DIM_model(batch_s=64,num_classes =128,feature=True)   
            dim_model.to(device)
            classifierM = classifier(n_input=128,n_class=10)
            classifierM = classifierM.to(device)
            writer = SummaryWriter('runs/experiment_C'+str(i))
            lr_new = 0.00001
            lrC=0.0001
            epC=50
        else:
            prior = True
            ep=8
            epC=10
            lr_new =0.000005
            lrC = 0.00005

        optimizer = torch.optim.Adam(dim_model.parameters(),lr=lr_new)
        scheduler = lr_scheduler.StepLR(optimizer,step_size=28,gamma=0.1) #there is also MultiStepLR
        tr_dict_enc = {'ep':ep,'writer':writerDIM,'best_loss':1e10,'t_board':True,'gamma':.5,'beta':.5,'Prior_Flag':prior,'discriminator':classifierM}    
        tr_dict_cl = {'ep':50,'writer':writer,'best_loss':1e10,'t_board':True,'gamma':1}

        if i==0 and load:
            print('Load DIM model weights first step')
            dim_model.load_state_dict(torch.load('/home/jbonato/Documents/cvpr_clvision_challenge/weights/weightsDIM_T0_NC_128.pt'))
        else:
            dim_model = trainEnc_MI(dim_model, optimizer, scheduler,dataloaders,device,tr_dict_enc)
            torch.save(dim_model.state_dict(), '/home/jbonato/Documents/cvpr_clvision_challenge/weights/weightsDIM_T'+str(i)+'_NC_128.pt')

        #if i==0:
        dataTr,labTr = save_prior_dist(dim_model,train_loader,device)
        dataCv,labCv = save_prior_dist(dim_model,valid_loader,device)

        print(dataTr.shape,labTr.shape)

        train_set = LoadFeatures(dataTr,labTr)
        val_set = LoadFeatures(dataCv,labCv)
        batch_size=32

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        dataloaderC = {'train':train_loader,'val':valid_loader}
        
        optimizerC = torch.optim.Adam(classifierM.parameters(),lr=lrC)
        schedulerC = lr_scheduler.StepLR(optimizerC,step_size=40,gamma=0.1)
        classifierM.requires_grad_(True)
        classifierM = train_classifier(classifierM, optimizerC, schedulerC,dataloaderC,device,tr_dict_cl)
        torch.save(classifierM.state_dict(), '/home/jbonato/Documents/cvpr_clvision_challenge/weights/weightsC_T'+str(i)+'_NC_128.pt')
        
        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())
    ######## print mem 
        print('Memory usage',np.asarray(stats['ram']).mean())
    
    #### test Parte on coreset to undestand performance
        for jj in range(len(test)):
            data_test = test[jj][0][0]
            labels_test = test[jj][0][1]
            task = test[jj][1]
            
            test_set = LoadDataset(data_test,labels_test,transform=None)
            batch_size=100
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
            score= []
            dim_model.eval()
            classifierM.eval()
            for inputs, labels in test_loader:
                torch.cuda.empty_cache()
                inputs = inputs.to(device)
                labels = labels.to(device) 
                _,_,ww =dim_model(inputs)
                pred = classifierM(ww)
                pred_l = pred.data.cpu().numpy()
                pred_l = np.argmax(pred_l,axis=1)
                out_lab = revert_lab(pred_l,task)
                score.append(np.sum(out_lab==labels.data.cpu().numpy())/out_lab.shape[0])
            print('TEST PERFORMANCES:', np.asarray(score).mean())
            del test_set,test_loader

else:
    dim_model = DIM_model(batch_s=32,num_classes =128,feature=True)   
    dim_model.to(device)
    classifierM = classifier(n_input = 128,n_class=10)
    classifierM = classifierM.to(device)
    print('Load DIM model weights first step')
    errors = np.zeros((50,))
    final = []
    for i in [8]:#range(8):
        final = []
        dim_model.load_state_dict(torch.load('/home/jbonato/Documents/cvpr_clvision_challenge/weights/weightsDIM_T'+str(i)+'_NC_128.pt'))    
        classifierM.load_state_dict(torch.load('/home/jbonato/Documents/cvpr_clvision_challenge/weights/weightsC_T'+str(i)+'_NC_128.pt'))
        for jj in range(len(test)):
            data_test = test[jj][0][0]
            labels_test = test[jj][0][1]
            task = test[jj][1]

            test_set = LoadDataset(data_test,labels_test,transform=None)
            batch_size=100
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
            score= []
            dim_model.eval()
            classifierM.eval()
            for inputs, labels in test_loader:
                torch.cuda.empty_cache()
                inputs = inputs.to(device)
                labels = labels.to(device) 
                _,_,ww =dim_model(inputs)
                pred = classifierM(ww)
                pred_l = pred.data.cpu().numpy()
                pred_l = np.argmax(pred_l,axis=1)
                out_lab = revert_lab(pred_l,task,case)
                score.append(np.sum(out_lab==labels.data.cpu().numpy())/out_lab.shape[0])
            print('TEST PERFORMANCES:', np.asarray(score).mean())
            final.append(np.asarray(score).mean())
        print(np.asarray(final).mean())
# print(errors)

# a,b = np.unique(labels_test,True)
# print(a)
# print(b)