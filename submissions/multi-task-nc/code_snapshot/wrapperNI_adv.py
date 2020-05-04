import argparse
import os
import time
import copy
import six
import sys

import numpy as np

#import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
#import torch.nn as nn
#import torch

import matplotlib.pyplot as plt

### tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

import torch
import numpy

from DIM_model import *
from train_nets import *
from PP_misc.loader import data_split,data_split_Tr_CV,LoadFeatures #LoadDataset
from PP_misc.transf import Transform
from PP_misc.cutout import Cutout
from model import _classifier

from train_prior_disc import save_prior_dist

import torch

# class LoadDataset(Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self,images,labels=None,transform=None,indices=None,ref=None):
#         """
#         Args:
#             images -> np.arr (samples,H,W,C)
#             labels -> np.arr (sample,)
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         if not(indices is None):
#             self.index = indices
#         else:
#             self.index = np.arange(images.shape[0])
            
#         self.im = images[self.index].astype(np.uint8)
#         if not(labels is None):
#             self.lb = labels[self.index]
#         else: 
#             self.lb = None
#         self.transform = transform
#         self.ref = ref
#         if not(self.ref is None):
#             self.standardize = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.60010594, 0.57207793, 0.54166424], [0.10679197, 0.10496728, 0.10731174])
#             ])
 
#     def __len__(self):
#         return len(self.im)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         image = self.im[idx]
#         #print(image.shape)
#         if self.transform:
#             image = self.transform(image)
#         if self.ref is None:    
#             if not(self.lb is None):
#                 label = self.lb[idx]
#                 return image,label
#             else:
#                 return image
#         else:
#             if not(self.lb is None):
#                 label = self.lb[idx]
#                 pt = np.where(self.ref[1]==label)
#                 idx_ref = np.random.choice(pt[0],1) 
#                 reference = self.ref[0][idx_ref,:,:,:] 
#                 #print(reference.shape,reference.dtype,self.ref[1][idx_ref])
#                 plt.imshow(reference[0].astype(np.uint8))
#                 return [image,self.standardize(reference[0])],label
#             else:
#                 return image

def data_org(data,lab):
    L = lab.max()
    gen_list = []
    for i in range(int(L)+1):
        pt = np.where(lab==i)
        gen_list.append(data[pt[0],:,:,:])
        #print(data[pt[0],:,:,:].shape)
    return gen_list

class LoadDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,images,labels=None,transform=None,indices=None,ref=None):
        """
        Args:
            images -> np.arr (samples,H,W,C)
            labels -> np.arr (sample,)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if not(indices is None):
            self.index = indices
        else:
            self.index = np.arange(images.shape[0])
            
        self.im = images[self.index].astype(np.uint8)
        if not(labels is None):
            self.lb = labels[self.index]
        else: 
            self.lb = None
        self.transform = transform
        self.ref = ref
        if not(self.ref is None):
            self.standardize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.60010594, 0.57207793, 0.54166424], [0.10679197, 0.10496728, 0.10731174])
            ])
 
    def __len__(self):
        return len(self.im)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.im[idx]
        #print(image.shape)
        if self.transform:
            image = self.transform(image)
        if self.ref is None:    
            if not(self.lb is None):
                label = self.lb[idx]
                return image,label
            else:
                return image
        else:
            if not(self.lb is None):
                label = self.lb[idx]
                #pt = np.where(self.ref[1]==label)
                #print(len(self.ref),int(label))
                vec_choice = self.ref[int(label)]
                idx_ref = np.random.randint(vec_choice.shape[0], size=1)#np.random.choice(np.arange(),1) 
                
                reference = vec_choice[idx_ref,:,:,:] 
                out = torch.empty((3,128,128,2),dtype=torch.float32)
                out[:,:,:,0]=image
                out[:,:,:,1]=self.standardize(reference[0])
                #print(out.size(),image.size())
                return out,label
            else:
                return image

class NI_wrap():
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
        self.trT = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.60010594, 0.57207793, 0.54166424], [0.10679197, 0.10496728, 0.10731174])
            ])
        #Transform(affine=0.5, train=True,cutout_ratio=0.6,ssr_ratio=0.6,flip = 0.6)
        self.device = device
        self.path = path 
        
    def train(self):
        acc_time = []
        data_test = self.val_data[0][0][0]
        labels_test = self.val_data[0][0][1]
        for i, train_batch in enumerate(self.dataset):
            
            writerDIM = SummaryWriter('runs/experiment_DIM'+str(i))
            data,labels, t = train_batch

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
            if i==0: 
                train_set = LoadDataset(dataC,labC,transform=self.tr,indices=trC)
                val_set = LoadDataset(dataC,labC,transform=self.tr,indices=cvC)
            else:
                #mod for previous batches MI
                dataR = data_org(dataP,labP)
                train_set = LoadDataset(dataC,labC,transform=self.tr,indices=trC,ref=dataR)
                val_set = LoadDataset(dataC,labC,transform=self.tr,indices=cvC,ref=dataR)

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
                
                classifierM = _classifier(n_input=128,n_class=50,n_neurons=[256,256,128])
                classifierM = classifierM.to(self.device)
                writer = SummaryWriter('runs/experiment_C'+str(i))
                lr_new = 0.00001
                lrC=0.0001
                #lrC=0.001
                epC=50
            else:
                prior = True
                ep=8
                epC=10
                lr_new =0.000005
                lrC = 0.00005
                #lrC = 0.0005

            optimizer = torch.optim.Adam(dim_model.parameters(),lr=lr_new)
            scheduler = lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1) #there is also MultiStepLR

            tr_dict_enc = {'ep':ep,'writer':writerDIM,'best_loss':1e10,'t_board':True,
                           'gamma':.5,'beta':.5,'Prior_Flag':prior,'discriminator':classifierM}    
            tr_dict_cl = {'ep':30,'writer':writer,'best_loss':1e10,'t_board':True,'gamma':1}

            if i==0 and self.load:
                print('Load DIM model weights first step')
                dim_model.load_state_dict(torch.load(self.path + 'weights/weightsDIM_T0cset128_cnn.pt'))
            else:
                ############################## Train Encoder########################################
                dim_model,self.stats = trainEnc_MI(self.stats,dim_model, optimizer, scheduler,dataloaders,self.device,tr_dict_enc)
                ####################################################################################
                torch.save(dim_model.state_dict(), self.path + 'weights/weightsDIM_T'+str(i)+'cset128_cnn.pt')

            #if i==0:
            #dataTr,labTr = save_prior_dist(dim_model,train_loader,self.device)
            #dataCv,labCv = save_prior_dist(dim_model,valid_loader,self.device)

            #print(dataTr.shape,labTr.shape)
            ###########
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



            train_set = LoadFeatures(dataTr_f,labTr_f)
            val_set = LoadFeatures(dataCv_f,labCv_f)
            batch_size=32

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
            dataloaderC = {'train':train_loader,'val':valid_loader}

            optimizerC = torch.optim.Adam(classifierM.parameters(),lr=lrC)
            #optimizerC = torch.optim.SGD(classifierM.parameters(),lr=lrC,momentum=0.8)
            schedulerC = lr_scheduler.StepLR(optimizerC,step_size=40,gamma=0.1)
            classifierM.requires_grad_(True)

            ############################## Train Classifier ########################################
            classifierM,self.stats = train_classifier(self.stats,classifierM, optimizerC, schedulerC,dataloaderC,self.device,tr_dict_cl)
            #################################### #################################### ##############

            torch.save(classifierM.state_dict(), self.path + 'weights/weightsC_T'+str(i)+'cset128_cnn.pt')
            dim_model.requires_grad_(True)
            #### Cross_val Testing

            test_set = LoadDataset(data_test,labels_test,transform=self.trT)
            batch_size=32
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
            score= []
            dim_model.eval()
            classifierM.eval()
            for inputs, labels in test_loader:
                torch.cuda.empty_cache()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device) 
                _,_,ww =dim_model(inputs)
                pred = classifierM(ww)
                pred_l = pred.data.cpu().numpy()
                score.append(np.sum(np.argmax(pred_l,axis=1)==labels.data.cpu().numpy())/pred_l.shape[0])
            print('TEST PERFORMANCES:', np.asarray(score).mean())
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
            
            self.classifierM = _classifier(n_input=128,n_class=50,n_neurons=[256,256,128])
            self.classifierM = self.classifierM.to(self.device)  
            
            self.dim_model.load_state_dict(torch.load(self.path + 'weights/weightsDIM_T7cset128_cnn.pt'))
            self.classifierM.load_state_dict(torch.load(self.path + 'weights/weightsC_T7cset128_cnn.pt'))

        
        test_set = LoadDataset(test_data[0][0][0],transform=self.trT)
        batch_size=32
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        score = None
        self.dim_model.eval()
        self.classifierM.eval()
        for inputs in test_loader:
            torch.cuda.empty_cache()
            inputs = inputs.to(self.device)
            _,_,ww =self.dim_model(inputs)
            pred = self.classifierM(ww)
            pred_l = pred.data.cpu().numpy()
            if score is None:
                score = np.argmax(pred_l,axis=1)
            else:
                score = np.concatenate((score,np.argmax(pred_l,axis=1)),axis=0)      
        return score

