import os
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset


def random_flip(data,p=0.5):
    #data must be [C,H,W]
    dims = len(np.shape(data))
    if dims==2:
        data = np.expand_dims(data,axis=0)
    
    if random.random() < p:
        data = np.flip(data,1)
    if random.random() < p:
        data = np.flip(data,2)
    if dims==2:
        return data[0]
    return data
    
def random_rot90(data,p=0.5):
    dims = len(np.shape(data))
    if dims==2:
        data = np.expand_dims(data,axis=0)
    
    k = np.random.choice((1,2,3))
    
    if random.random() < p:
        data = np.rot90(data,k,axes=(1,2))
       
    if dims==2:
        return data[0]
    return data







class ACDCTrainDataset(Dataset):
    def __init__(self, base_dir,flip=True,rot=True):
        self._base_dir = base_dir
        self.sample_list = glob(base_dir+'/slice/*.npy')
        self.flip = flip
        self.rot = rot

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        data = np.load(self.sample_list[idx])
        if self.flip:
            data = random_flip(data)
        if self.rot:
            data = random_rot90(data)
        image = data[0:1]
        label = data[1:]
        image = torch.from_numpy(image.copy())
        label = torch.from_numpy(label.copy())
                
        return image,label