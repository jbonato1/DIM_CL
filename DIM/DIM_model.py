import torch.nn as nn
import torch
import numpy

from DIM.model import *
from DIM.mi_networks import *


class DIM_model(nn.Module):
    def __init__(self,batch_s = 32):
        super().__init__()
        # as a starting encoder we use alexnet 
        # we can try other architecture such as resNet, ecc., 
        # also pretrained version of the encoder could be a good option
        # these are exp to do
        
        self.encoder = alexnet(num_classes = 256)
        
        #test input output size and channel to use
        fake_in = torch.ones([2,3,128,128])
        out1,out2 = self.encoder(fake_in)
        n_input = out1.size(1)
        n_units = 2048
        
        # insert in the model mutual information networks
        self.global_MI = Local_MI_Gl_Feat(n_input = n_input,n_units = n_units)
        self.local_MI = Local_MI_1x1ConvNet(n_input,n_units)
        self.batch = batch_s
        self.features_g = n_units
        self.features_l = n_units

    def forward(self,x):
        
        E_phi, C_phi = self.encoder(x)
        E_phi = self.global_MI(E_phi)
        C_phi = self.local_MI(C_phi)
        E_phi = E_phi.view(self.batch,self.features_g,1)
        C_phi = C_phi.view(self.batch,self.features_l,-1)
        return E_phi,C_phi