import argparse
import os
import time
import copy
import six
import sys

import numpy as np

#import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
#import torch.nn as nn
#import torch
from torchvision import transforms

import matplotlib.pyplot as plt

### tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

import torch
import numpy

from networks.DIM_model import *
from networks.train_nets import *
from pre_proc.loader import LoadDataset,data_split,data_split_Tr_CV,LoadFeat
from pre_proc.transf import Transform 
from networks.model import classifier

from networks.train_prior_disc import save_prior_dist

class NC_wrap():
    def __init__(self,dataset,val_data,device,path,load=False,replay=True):
        '''
        Args:
        TO DO: complete Args
        
        '''
        self.load = load
        self.replay = replay
        self.stats = {"ram": [], "disk": []}
        self.dataset = dataset
        self.val_data = val_data
        
        self.tr = transforms.Compose([
    
            transforms.ToPILImage(),
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=0.6),
                transforms.ColorJitter(contrast=0.4),
                transforms.ColorJitter(saturation=0.4),
                ]),
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
                transforms.RandomRotation(180, resample=3, expand=False, center=None, fill=0),
                transforms.RandomAffine(30, translate=(.1,.1), scale=(0.95,1.05), shear=5, resample=False, fillcolor=0)
            ]),

            transforms.ToTensor(),
            #Cutout(4,20,p=0.6),
            transforms.Normalize([0.60010594, 0.57207793, 0.54166424], [0.10679197, 0.10496728, 0.10731174])
            ])
        ################## transformation for validation and test set
        self.trT = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.60010594, 0.57207793, 0.54166424], [0.10679197, 0.10496728, 0.10731174])
            ])
        self.device = device
        self.path = path 
        self.map_lb = {'0':None,'1':'B','2':'A','3':'B','4':'A','5':'A','6':'A','7':'B','8':'B'}
        
        
    def convert_lab(self,labels,task):
        if task!=0:
            case = self.map_lb[str(task)]
            print(case)
            if case == 'A':
                n_labels = labels%5
            if case =='B':
                n_labels = labels%5 + 5
            return n_labels
        else:
            return labels
        
    def revert_lab(self,labels,task):
        if task!=0:
            case = self.map_lb[str(task)]
            if case == 'A':
                n_labels = labels+task*5+5
            if case == 'B':
                n_labels = labels+task*5
            return n_labels
        else:
            return labels

    def train(self):
        acc_time = []

        for i, train_batch in enumerate(self.dataset):
            
            writerDIM = SummaryWriter('runs/experiment_DIM'+str(i))
            data,labelsI, t = train_batch
            labels = self.convert_lab(labelsI,t)            
            index_tr,index_cv,coreset = data_split(data.shape[0],777)

            # adding eventual replay patterns to the current batch
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
                if self.replay:
                    dataC = np.concatenate((data[index_tr], data[index_cv],dataP),axis=0)
                    labC = np.concatenate((labels[index_tr],labels[index_cv],labP),axis=0)
                else:
                    dataC = np.concatenate((data[index_tr], data[index_cv]),axis=0)
                    labC = np.concatenate((labels[index_tr],labels[index_cv]),axis=0)



            print("----------- batch {0} -------------".format(i))
            print("Task Label: ", t)
            trC,cvC = data_split_Tr_CV(dataC.shape[0],777)

            train_set = LoadDataset(dataC,labC,transform=self.tr,indices=trC)
            val_set = LoadDataset(dataC,labC,transform=self.tr,indices=cvC)
            print('Training set: {0} \nValidation Set {1}'.format(train_set.__len__(),val_set.__len__()))
            batch_size=32
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
            dataloaders = {'train':train_loader,'val':valid_loader}

            if i ==0:        
                prior = False
                ep=40
                dim_model = DIM_model(batch_s=32,num_classes =128,feature=True)   
                dim_model.to(self.device)
                classifierM = classifier(n_input=128,n_class=50)
                classifierM = classifierM.to(self.device)
                writer = SummaryWriter('runs/experiment_C'+str(i))
                lr_new = 0.00001
                lrC=0.0001
                epC=50
            else:
                prior = True
                ep=6
                epC=10
                lr_new =0.000005
                lrC = 0.00005

            optimizer = torch.optim.Adam(dim_model.parameters(),lr=lr_new)
            scheduler = lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.1) #there is also MultiStepLR

            tr_dict_enc = {'ep':ep,'writer':writerDIM,'best_loss':1e10,'t_board':True,
                           'gamma':.5,'beta':.5,'Prior_Flag':prior,'discriminator':classifierM}    
            tr_dict_cl = {'ep':30,'writer':writer,'best_loss':1e10,'t_board':True,'gamma':1}#30

            if i==0 and self.load:
                print('Load DIM model weights first step')
                dim_model.load_state_dict(torch.load(self.path + 'weights/weightsDIM_T0_NC128.pt'))
            else:
                ############################## Train Encoder########################################
                dim_model,self.stats = trainEnc_MI(self.stats,dim_model, optimizer, scheduler,dataloaders,self.device,tr_dict_enc)
                ####################################################################################
                torch.save(dim_model.state_dict(), self.path + 'weights/weightsDIM_T'+str(i)+'_NC.pt')

            #if i==0:
#             dataTr,labTr = save_prior_dist(dim_model,train_loader,self.device)
#             dataCv,labCv = save_prior_dist(dim_model,valid_loader,self.device)
            dim_model.requires_grad_(False)
            for phase in ['train','val']:
                dataF = None
                labF = None
                for inputs, labels in dataloaders[phase]:
                    torch.cuda.empty_cache()
                    if len(inputs.shape)==5:

                        inputs = inputs[:,:,:,:,0].to(self.device)
                    else:
                        inputs = inputs.to(self.device)

                    _,_,pred = dim_model(inputs)
                    pred_l = pred.data.cpu().numpy()
                    if dataF is None:
                        dataF = pred_l
                        labF = labels.data.cpu().numpy()
                    else:
                        dataF = np.concatenate((dataF,pred_l),axis=0)
                        labF = np.concatenate((labF,labels.data.cpu().numpy()),axis=0)

                if phase == 'train':
                    dataTr_f = dataF
                    labTr_f  = labF
                else:
                    dataCv_f = dataF
                    labCv_f = labF
                
            dim_model.requires_grad_(True)
            train_set = LoadFeat(dataTr_f,labTr_f)
            val_set = LoadFeat(dataCv_f,labCv_f)
            batch_size=32

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
            dataloaderC = {'train':train_loader,'val':valid_loader}

            optimizerC = torch.optim.Adam(classifierM.parameters(),lr=lrC)
            schedulerC = lr_scheduler.StepLR(optimizerC,step_size=40,gamma=0.1)
            classifierM.requires_grad_(True)

            ############################## Train Classifier ########################################
            classifierM,self.stats = train_classifier(self.stats,classifierM, optimizerC, schedulerC,dataloaderC,self.device,tr_dict_cl)
            #################################### #################################### ##############

            torch.save(classifierM.state_dict(), self.path + 'weights/weightsC_T'+str(i)+'_NC.pt')
            
            #### Cross_val Testing
            score = []
            for task_i in range(len(self.val_data)):
                
                data_test = self.val_data[task_i][0][0]
                labels_test = self.val_data[task_i][0][1]
                task = task_i

                test_set = LoadDataset(data_test,labels_test,transform=self.trT)
                batch_size=32
                test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
                score_t= []
                dim_model.eval()
                classifierM.eval()
                for inputs, labels in test_loader:
                    torch.cuda.empty_cache()
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device) 
                    _,_,ww =dim_model(inputs)
                    pred = classifierM(ww)
                    pred_l = pred.data.cpu().numpy()
                    pred_l = np.argmax(pred_l,axis=1)
                    out_lab = self.revert_lab(pred_l,task)
                    score_t.append(np.sum(out_lab==labels.data.cpu().numpy())/out_lab.shape[0])
                    
                print('TEST PERFORMANCES:', np.asarray(score_t).mean())
                score.append(np.asarray(score_t).mean())
            acc_time.append(np.asarray(score).mean())
            del test_set,test_loader
            
        self.dim_model = dim_model
        self.classifierM = classifierM
        acc_time = np.asarray(acc_time)
        return self.stats,acc_time
        
    def test(self,test_data,standalone=False):
        
        if standalone:
            self.dim_model = DIM_model(batch_s=32,num_classes =128,feature=True)   
            self.dim_model.to(self.device)
            
            self.classifierM = classifier(n_input = 128,n_class=50)
            self.classifierM = self.classifierM.to(self.device)  
            
            self.dim_model.load_state_dict(torch.load(self.path + 'weights/weightsDIM_T0_NC.pt'))
            self.classifierM.load_state_dict(torch.load(self.path + 'weights/weightsC_T0_NC.pt'))

        
        test_set = LoadDataset(test_data[0][0][0],transform=None)
        batch_size=100
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        score= []
        self.dim_model.eval()
        self.classifierM.eval()
        for inputs in test_loader:
            torch.cuda.empty_cache()
            inputs = inputs.to(self.device)
            _,_,ww =self.dim_model(inputs)
            pred = self.classifierM(ww)
            pred_l = pred.data.cpu().numpy()
            pred_l = np.argmax(pred_l,axis=1)
            out_lab = self.revert_lab(pred_l,task)
            if score is None:
                score = out_lab
            else:
                score = np.concatenate((score,out_lab),axis=0)      
        return score
