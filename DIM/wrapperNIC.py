import os
import time
import copy
import six
import sys
import numpy as np
from torch.utils.data.dataloader import DataLoader


### tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import torch


from networks.DIM_model import *
from networks.train_nets import *
from pre_proc.loader import LoadDataset,data_split,data_split_Tr_CV,LoadFeatures
from pre_proc.transf import Transform 
from networks.model import _classifier
from networks.train_prior_disc import save_prior_dist

class NIC_wrap():
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
        self.device = device
        self.path = path 
        
    def train(self):
        acc_time = [0]
        data_test = self.val_data[0][0][0]
        labels_test = self.val_data[0][0][1]
        labels_seen = []
        num = 0
        data_cur = None
        for i, train_batch in enumerate(self.dataset):
            store =False
            data,labels, t = train_batch
            print(labels_seen,i,np.unique(labels))

            lb =np.unique(labels).astype(np.int64).tolist()
            for l in lb:
                if l in labels_seen:
                    store = True
                else:
                    labels_seen.append(l)

            if store and i!=390:
                print('store')
                if data_cur is None:
                    data_cur = data
                    labels_cur = labels
                else:
                    data_cur =np.concatenate((data_cur,data),axis=0)
                    labels_cur =np.concatenate((labels_cur,labels),axis=0)
            elif (not(store) and i!=0) or i==390:
                print('new Train')
                if i==390:
                    store=False

                if data_cur is None:
                    data_cur = data
                    labels_cur = labels
                else:
                    data_cur =np.concatenate((data_cur,data),axis=0)
                    labels_cur =np.concatenate((labels_cur,labels),axis=0)

                ### extract cur_replay 
                index_tr,index_cv,coreset = data_split(data_cur.shape[0],777)
                ### add previous replay
                if self.replay:
                    dataP = ext_mem[0]
                    labP = ext_mem[1]
                    dataC = np.concatenate((data_cur[index_tr], data_cur[index_cv],dataP),axis=0)
                    labC = np.concatenate((labels_cur[index_tr],labels_cur[index_cv],labP),axis=0)
                else:
                    dataC = np.concatenate((data_cur[index_tr], data_cur[index_cv]),axis=0)
                    labC = np.concatenate((labels_cur[index_tr],labels_cur[index_cv]),axis=0)

                ### merge replay with cur_replay
                ext_mem = [
                    np.concatenate((data_cur[coreset], ext_mem[0])),
                    np.concatenate((labels_cur[coreset], ext_mem[1]))]
                data_cur = None
                del labels_cur

            else:
                index_tr,index_cv,coreset = data_split(data.shape[0],777)
                ext_mem = [data[coreset], labels[coreset]]
                dataC = np.concatenate((data[index_tr], data[index_cv]),axis=0)
                labC = np.concatenate((labels[index_tr],labels[index_cv]),axis=0)
                
            if not(store) or i==0:#>=390:
                writerDIM = SummaryWriter(self.path+'runs/experiment_DIM'+str(i))
                print("----------- batch {0} -------------".format(i))
                trC,cvC = data_split_Tr_CV(dataC.shape[0],777) 
                train_set = LoadDataset(dataC,labC,transform=self.tr,indices=trC)
                val_set = LoadDataset(dataC,labC,transform=self.tr,indices=cvC)
                print('Training set: {0} \n Validation Set {1}'.format(train_set.__len__(),val_set.__len__()))
                batch_size=32
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
                valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
                dataloaders = {'train':train_loader,'val':valid_loader}

                if i ==0:        
                    prior = False
                    ep=30
                    dim_model = DIM_model(batch_s=32,num_classes =128,feature=True)   
                    dim_model.to(self.device)
                    classifierM = _classifier(n_input=128,n_class=50,n_neurons=[512,256,128])
                    classifierM = classifierM.to(self.device)
                    writer = SummaryWriter(self.path+'runs/experiment_C'+str(i))
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
                scheduler = lr_scheduler.StepLR(optimizer,step_size=25,gamma=0.1) #there is also MultiStepLR
                tr_dict_enc = {'ep':ep,'writer':writerDIM,'best_loss':1e10,'t_board':True,'gamma':.5,'beta':.5,
                               'Prior_Flag':prior,'discriminator':classifierM}    
                tr_dict_cl = {'ep':50,'writer':writer,'best_loss':1e10,'t_board':True,'gamma':1}

                if i==390 and self.load==True:
                    print('Load DIM model weights first step')
                    dim_model.load_state_dict(torch.load(self.path+'weights/weightsDIM_T340_nic.pt'))
                else:
                    ############################## Train Encoder########################################
                    dim_model,self.stats = trainEnc_MI(self.stats,dim_model, optimizer, scheduler,dataloaders,self.device,tr_dict_enc)
                     ############################## ############################## ##############################
                    if i ==340:    
                        torch.save(dim_model.state_dict(),self.path+ 'weights/weightsDIM_T'+str(i)+'_nic.pt')

                
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
                if i ==340:
                    torch.save(classifierM.state_dict(), '/home/jbonato/Documents/cvpr_clvision_challenge/weights/weightsC_T'+str(i)+'_nic.pt')

                #### Validation Set Performances
                
                test_set = LoadDataset(data_test,labels_test,transform=None)
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
            else:
                acc_time.append(acc_time[-1])
                
        self.dim_model = dim_model
        self.classifierM = classifierM
        acc_time = np.asarray(acc_time)
        return self.stats,acc_time
                

            
        
    def test(self,test_data,standalone=False):
        
        if standalone:
            self.dim_model = DIM_model(batch_s=32,num_classes =128,feature=True)   
            self.dim_model.to(self.device)
            
            self.classifierM = _classifier(n_input=128,n_class=50,n_neurons=[512,256,128])
            self.classifierM = self.classifierM.to(self.device)  
            
            self.dim_model.load_state_dict(torch.load(self.path + 'weights/weightsDIM_T0cset128.pt'))
            self.classifierM.load_state_dict(torch.load(self.path + 'weights/weightsC_T0cset128.pt'))

        
        test_set = LoadDataset(test_data[0][0][0],transform=None)
        batch_size=100
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
