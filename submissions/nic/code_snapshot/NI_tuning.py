from ray import tune
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

import matplotlib.pyplot as plt

### tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

import torch
import numpy

from DIM_model import *
from train_nets import *
from PP_misc.loader import LoadDataset,data_split,data_split_Tr_CV,LoadFeatures
from PP_misc.transf import Transform 
from model import classifier

from train_prior_disc import save_prior_dist

sys.path.append('/home/jbonato/Documents/cvpr_clvision_challenge/')
from core50.dataset import CORE50

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
        
        self.tr = Transform(affine=0.5, train=True,cutout_ratio=0.6,ssr_ratio=0.6,flip = 0.6)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        #self.path = path 
        
    def train(self,feat_in,neuron_list):
        acc_time = []
        data_test = self.val_data[0][0][0]
        labels_test = self.val_data[0][0][1]
        for i, train_batch in enumerate(self.dataset):
            
            writerDIM = None#SummaryWriter('runs/experiment_DIM'+str(i))
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

            train_set = LoadDataset(dataC,labC,transform=self.tr,indices=trC)
            val_set = LoadDataset(dataC,labC,transform=self.tr,indices=cvC)
            print('Training set: {0} \nValidation Set {1}'.format(train_set.__len__(),val_set.__len__()))
            batch_size=32
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
            dataloaders = {'train':train_loader,'val':valid_loader}

            if i ==0:        
                prior = False
                ep=80
                dim_model = DIM_model(batch_s=32,num_classes =feat_in,feature=True)   
                dim_model.to(self.device)
                classifierM = _classifier(n_input=feat_in,n_class=50,n_neurons=neuron_list)
                classifierM = classifierM.to(self.device)
                writer = None#SummaryWriter('runs/experiment_C'+str(i))
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
            scheduler = lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.1) #there is also MultiStepLR

            tr_dict_enc = {'ep':ep,'writer':writerDIM,'best_loss':1e10,'t_board':False,
                           'gamma':.5,'beta':.5,'Prior_Flag':prior,'discriminator':classifierM}    
            tr_dict_cl = {'ep':50,'writer':writer,'best_loss':1e10,'t_board':False,'gamma':1}

            if i==0 and self.load:
                print('Load DIM model weights first step')
                #dim_model.load_state_dict(torch.load(self.path + 'weights/weightsDIM_T0cset128.pt'))
            else:
                ############################## Train Encoder########################################
                dim_model,self.stats = trainEnc_MI(self.stats,dim_model, optimizer, scheduler,dataloaders,self.device,tr_dict_enc)
                ####################################################################################
                #torch.save(dim_model.state_dict(), self.path + 'weights/weightsDIM_T'+str(i)+'cset128.pt')

            #if i==0:
            dataTr,labTr = save_prior_dist(dim_model,train_loader,self.device)
            dataCv,labCv = save_prior_dist(dim_model,valid_loader,self.device)

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

            ############################## Train Classifier ########################################
            classifierM,self.stats = train_classifier(self.stats,classifierM, optimizerC, schedulerC,dataloaderC,self.device,tr_dict_cl)
            #################################### #################################### ##############

            #torch.save(classifierM.state_dict(), self.path + 'weights/weightsC_T'+str(i)+'cset128.pt')

            #### Cross_val Testing

            test_set = LoadDataset(data_test,labels_test,transform=None)
            batch_size=100
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
            #print('TEST PERFORMANCES:', np.asarray(score).mean())
            acc_time.append(np.asarray(score).mean())
            del test_set,test_loader
        #self.dim_model = dim_model
        #self.classifierM = classifierM
        acc_time = np.asarray(acc_time)
        return acc_time[-1]

    def train_config(self,config):
        feat_in = config["feature_in"]
        neuron_list = config["neuron_list"]
        acc = self.train(feat_in,neuron_list)
        tune.track.log(mean_accuracy=acc)

dataset = CORE50(root='/home/jbonato/Documents/cvpr_clvision_challenge/core50/data/', scenario='multi-task-nc',preload=True)
test =  dataset.get_full_valid_set(reduced=True)

NI = NI_wrap(dataset,test,load=False,replay=True):

space={'feature_in'=tune.choioce([64,128,256,512]),
      'neuron_list'=tune.choice([[512,256,128],[512,512],[512,256,256]])
      }

analysis = tune.run(NI.train_config,config=space,resources_per_trial={"cpu": 4,"gpu": 3})

print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()