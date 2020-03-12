import numpy
import six
import torch
from torch.utils.data.dataset import Dataset
#import albumentations as A
import numpy as np
#import cv2
#from skimage.transform import AffineTransform, warp
import numpy as np
#import pandas as pd
import gc

class DatasetMixin(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
       

    def __getitem__(self, index):
        """Returns an example or a sequence of examples."""
        if torch.is_tensor(index):
            index = index.tolist()
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example_wrapper(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, numpy.ndarray):
            return [self.get_example_wrapper(i) for i in index]
        else:
            return self.get_example_wrapper(index)

    def __len__(self):
        """Returns the number of data points."""
        raise NotImplementedError

    def get_example_wrapper(self, i):
        """Wrapper of `get_example`, to apply `transform` if necessary"""
        example = self.get_example(i)
        
        if self.transform:
            example = self.transform(example)
   
        return example

    def get_example(self, i):
        """Returns the i-th example.

        Implementations should override it. It should raise :class:`IndexError`
        if the index is invalid.

        Args:
            i (int): The index of the example.

        Returns:
            The i-th example.

        """
        raise NotImplementedError

class LoadDataset(DatasetMixin):
    def __init__(self, images, labels=None, transform=None, indices=None):
        super(LoadDataset, self).__init__(transform=transform)
        self.images = images
        self.labels = labels
        if indices is None:
            indices = np.arange(len(images))
        self.indices = indices
        self.train = labels is not None

    def __len__(self):
        """return length of this dataset"""
        return len(self.indices)

    def get_example(self, i):
        """Return i-th data"""
        i = self.indices[i]
        x = self.images[i]
        #print(x.shape)
        # scale to [0,1] interval
        x = x/255
        
        # normalize
        x[ :, :, 0] = ((x[ :, :, 0] - 0.485) / 0.229)
        x[ :, :, 1] = ((x[ :, :, 1] - 0.456) / 0.224)
        x[ :, :, 2] = ((x[ :, :, 2] - 0.406) / 0.225)
        
        # Swap channel dimension to fit the caffe format (c, w, h)
        x = np.transpose(x, (2, 0, 1))        
        if self.train:
            y = self.labels[i]
            return x, y
        else:
            return x
        
class LoadFeatures(DatasetMixin):
    def __init__(self, features, labels=None, transform=None, indices=None):
        super(LoadFeatures, self).__init__(transform=transform)
        self.images = features
        self.labels = labels
        if indices is None:
            indices = np.arange(len(features))
        self.indices = indices
        self.train = labels is not None

    def __len__(self):
        """return length of this dataset"""
        return len(self.indices)

    def get_example(self, i):
        """Return i-th data"""
        i = self.indices[i]
        x = self.images[i]    
        if self.train:
            y = self.labels[i]
            return x, y
        else:
            return x
        
def data_split(n_dataset,seed,debug=False):
    '''This module divide data_batchin:
        90%(training, cv) and 10%(coreset)
        Args:
        -data
        -labels
        -seed: for splitting
        return:
        training, cv and coreset
    '''
    
    train_data_size = 10 if debug else int(n_dataset * 0.9)
    tr = 10 if debug else int(train_data_size * 0.8)
    cv = 10 if debug else int(train_data_size * 0.2)   
    core = 10 if debug else int(n_dataset - train_data_size)
 
    perm = np.random.RandomState(seed).permutation(n_dataset)
    
    return perm[core:core+tr],perm[core+tr:],perm[:core]
                                                  
def data_split_Tr_CV(train_data_size,seed,debug=False):
    '''This module divide data_batchin:
        80%(training, cv) and 20%(coreset)
        Args:
        -data
        -labels
        -seed: for splitting
        return:
        training, cv 
    '''
    tr = 10 if debug else int(train_data_size * 0.8)
    cv = 10 if debug else int(train_data_size * 0.2)   
   
 
    perm = np.random.RandomState(seed).permutation(train_data_size)
    
    return perm[:tr],perm[tr:]