"""

Getting Started example for the CVPR 2020 CLVision Challenge. It will load the
data and create the submission file for you in the
cvpr_clvision_challenge/submissions directory.

"""

import argparse
import os
import time
import copy
import torch
import numpy as np

import sys 
####
<<<<<<< HEAD
sys.path.append('/home/jbonato/Documents/cvpr_clvision_challenge/')
from core50.dataset import CORE50
from utils.common import create_code_snapshot
from DIM.wrapperNI import *
from DIM.wrapperNC import *
=======
sys.path.append('/media/DATA/jbonato/cvpr_clvision_challenge/')
from core50.dataset import CORE50
from utils.common import create_code_snapshot
from wrapperNI import *
from wrapperNC import *
from wrapperNIC import *
>>>>>>> fc9dbda0e6b2bce4d095cf0e5f5413e1e2c30199

def main(args):

    # print args recap
    print(args, end="\n\n")

    # do not remove this line
    start = time.time()

    # Create the dataset object for example with the "ni, multi-task-nc, or nic
    # tracks" and assuming the core50 location in ./core50/data/
<<<<<<< HEAD
    dataset = CORE50(root='/home/jbonato/Documents/cvpr_clvision_challenge/core50/data/', scenario=args.scenario,
=======
    dataset = CORE50(root='/media/DATA/jbonato/cvpr_clvision_challenge/core50/data/', scenario=args.scenario,
>>>>>>> fc9dbda0e6b2bce4d095cf0e5f5413e1e2c30199
                     preload=True)

    # Get the validation set
    print("Recovering validation set...")
<<<<<<< HEAD
    full_valdidset = dataset.get_full_valid_set()
    device0 = torch.device('cuda:0')
    # code for training 
    if args.scenario=='ni':
        NI = NI_wrap(dataset,full_valdidset,device=device0,path='/home/jbonato/Documents/cvpr_clvision_challenge/',load=True)
    elif args.scenario=='multi-task-nc':
        NI = NC_wrap(dataset,full_valdidset,device=device0,path='/home/jbonato/Documents/cvpr_clvision_challenge/',load=False)
    elif args.scenario=='nic':
        raise NotImplementedError
=======
    full_valdidset = dataset.get_full_valid_set(reduced=True)
    torch.cuda.set_device(int(args.gpu))
    device0 = torch.device('cuda:'+args.gpu)
    # code for training 
    if args.scenario=='ni':
        NI = NI_wrap(dataset,full_valdidset,device=device0,path='/media/DATA/jbonato/cvpr_clvision_challenge/',load=False)
    elif args.scenario=='multi-task-nc':
        NI = NC_wrap(dataset,full_valdidset,device=device0,path='/media/DATA/jbonato/cvpr_clvision_challenge/',load=False)
    elif args.scenario=='nic':
        NI = NIC_wrap(dataset,full_valdidset,device=device0,path='/media/DATA/jbonato/cvpr_clvision_challenge/',load=False)
>>>>>>> fc9dbda0e6b2bce4d095cf0e5f5413e1e2c30199
    stats,valid_acc = NI.train()
    ram_usage = np.asarray(stats['ram'])
    ext_mem_sz = np.asarray(stats['disk'])
    # Generate submission.zip
    # directory with the code snapshot to generate the results
    sub_dir = 'submissions/' + args.sub_dir
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # copy code
    create_code_snapshot(".", sub_dir + "/code_snapshot")

    # generating metadata.txt: with all the data used for the CLScore
    elapsed = (time.time() - start) / 60
    print("Training Time: {}m".format(elapsed))
    with open(sub_dir + "/metadata.txt", "w") as wf:
        for obj in [
            np.average(valid_acc), elapsed, np.average(ram_usage),
            np.max(ram_usage), np.average(ext_mem_sz), np.max(ext_mem_sz)
        ]:
            wf.write(str(obj) + "\n")

    # test_preds.txt: with a list of labels separated by "\n"
    print("Final inference on test set...")
    full_testset = dataset.get_full_test_set()

    pred = NI.test(full_testset,standalone=False)
    
    with open(sub_dir + "/test_preds.txt", "w") as wf:
        for jj in range(pred.shape[0]):
            wf.write(str(pred[jj]) + "\n")

    print("Experiment completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')

    # General
    parser.add_argument('--scenario', type=str, default="ni",
                        choices=['ni', 'multi-task-nc', 'nic'])
    parser.add_argument('--preload_data', type=bool, default=True,
                        help='preload data into RAM')
    
    parser.add_argument('--sub_dir', type=str, default="ni",
                        choices=['ni', 'multi-task-nc', 'nic'],
                        help='directory of the submission file for this exp.')

<<<<<<< HEAD
    
=======
    parser.add_argument('--gpu', type=str, default="0",
                        choices=['0','1','2'],
                        help='GPU_num')
>>>>>>> fc9dbda0e6b2bce4d095cf0e5f5413e1e2c30199
    args = parser.parse_args()
    main(args)
