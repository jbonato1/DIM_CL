import time
import copy
import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
import torch
import math

import torch.nn.functional as F
from torch.nn import Sequential
from torch.nn import init
from torch.nn.parameter import Parameter

###
from dim_loss import *
from train_prior_disc import train_disc,sample_prior
<<<<<<< HEAD
=======
import sys
sys.path.append('../')
>>>>>>> fc9dbda0e6b2bce4d095cf0e5f5413e1e2c30199
from func.common import check_ext_mem, check_ram_usage

def print_metrics(metrics, batch_num, phase):    
    """ Utility function to print dictionary elements that track loss ecc."""
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / batch_num))
        
    print("{}: {}".format(phase, ", ".join(outputs))) 

def update_metrics(loss,metrics):
    """Simple function to update metrics dictionary, Adding kargws it is simple to exted it to multiple 
        entries
    """
    metrics['loss'] += loss.data.cpu().numpy()
    

def compute_loss(pred,y,metrics):
    """Compute accuracy and cross entropy loss for classifier"""
    loss = F.cross_entropy(pred,y)
    metrics['loss'] += loss.data.cpu().numpy()    
    pred_l = pred.data.cpu().numpy()
    metrics['accuracy'] +=np.sum(np.argmax(pred_l,axis=1)==y.data.cpu().numpy())/pred_l.shape[0] 
    return loss
    
    
#### Eventually think about a generale trainer class

def trainEnc_MI(stats,model, optimizer, scheduler,dataloaders,device,kwargs):
    """This funcion performs training of DIM local version without prior distrib. It can be wrapped into the DNN
       module as a class function  
    """
    num_epochs=kwargs['ep']
    writer=kwargs['writer']
    best_loss=kwargs['best_loss']
    t_board=kwargs['t_board']
    gamma = kwargs['gamma']
    beta = kwargs['beta']
    use_prior = kwargs['Prior_Flag'] 
    discriminator = kwargs['discriminator'] 
    
    gen_epoch = 0
    gen_epoch_l=0
    
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            
            if use_prior:
                discriminator.requires_grad_(False)
                
            if phase == 'train':    
                model.train()  # Set model to training mode                
            else:
                model.eval()   # Set model to evaluate mode


            metrics = defaultdict(float)
            epoch_samples = 0
            batch_num = 0
            for inputs, labels in dataloaders[phase]:
                labels = labels.type(torch.long)
                torch.cuda.empty_cache()
                inputs = inputs.to(device)
                labels = labels.to(device)        
                optimizer.zero_grad()
                
                # forward
                with torch.set_grad_enabled(phase == 'train'):                    
                    E_phi,C_phi,A_phi = model(inputs)
                    loss = 0   
                    #we use jsd MI approx. since it is more stable eventually for other exp call compute dim loss
                    #function already implemented
                    loss_MI = fenchel_dual_loss(C_phi, E_phi, measure='JSD')
                    
<<<<<<< HEAD
=======
                    #infonce_loss(C_phi, E_phi) we need to test also this measure
                    
>>>>>>> fc9dbda0e6b2bce4d095cf0e5f5413e1e2c30199
                    if use_prior:
                        loss += beta*loss_MI
                    else:
                        loss = loss_MI
                    metrics['MI_loss'] += loss.data.cpu().numpy()                    
                    ################################## section for prior Loss
                    if use_prior:
                        Q_samples = discriminator(A_phi)
                        disc_loss = compute_loss(Q_samples,labels,metrics)
                        loss += gamma*disc_loss
                        
                    metrics['loss'] += loss.data.cpu().numpy()
                    ################################## backward and tensorboard: othre quantities can be added
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if batch_num % 20 == 0  and t_board:    
                            # ...log the running loss
                            writer.add_scalar('training loss',loss,epoch*gen_epoch+batch_num)
                    if phase == 'val' and t_board:
                        if batch_num % 10 == 0:    
                            # ...log the running loss
                            writer.add_scalar('Validation loss',loss,epoch*gen_epoch_l+batch_num)

                # statistics
                batch_num +=1
            ############ end of epochs
            
            epoch_loss = metrics['loss']/batch_num
            
            if epoch ==0 and phase=='train':
                gen_epoch = batch_num
            if epoch ==0 and phase=='val':
                gen_epoch_l = batch_num
                
            if phase == 'train':
                avg = metrics['loss']/batch_num
                scheduler.step()#avg

                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
            #### Write and Save
            print_metrics(metrics, batch_num, phase)
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        
        ############## Time Info
        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    del best_model_wts
    return model,stats

def train_classifier(stats,model, optimizer, scheduler,dataloaders,device,kwargs):
    """This funcion performs training of classifier HEAD. It can be wrapped into the DNN
       module as a class function  
    """
    
    num_epochs=kwargs['ep']
    writer=kwargs['writer']
    best_loss=kwargs['best_loss']
    t_board=kwargs['t_board']
    
    gen_epoch = 0
    gen_epoch_l=0
    
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        stats['disk'].append(check_ext_mem("cl_ext_mem"))
        stats['ram'].append(check_ram_usage())
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            batch_num = 0
            for inputs, labels in dataloaders[phase]:
                labels = labels.type(torch.long)
                torch.cuda.empty_cache()
                inputs = inputs.to(device)
                labels = labels.to(device)        
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):                    
                    
                    out = model(inputs)
                    loss = 0   
                    loss = compute_loss(out,labels,metrics)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if batch_num % 20 == 0  and t_board:    
                            writer.add_scalar('training loss',loss,epoch*gen_epoch+batch_num)
                            
                    if phase == 'val' and t_board:
                        if batch_num % 10 == 0:    
                            writer.add_scalar('Validation loss',loss,epoch*gen_epoch_l+batch_num)

                # statistics
                batch_num +=1
            epoch_loss = metrics['loss']/batch_num
            
            if epoch ==0 and phase=='train':
                gen_epoch = batch_num
            if epoch ==0 and phase=='val':
                gen_epoch_l = batch_num
            if phase == 'train':
                scheduler.step()#avg
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
            
            
            print_metrics(metrics, batch_num, phase)
            
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since

        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    del best_model_wts
    return model,stats