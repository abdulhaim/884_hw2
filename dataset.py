import argparse
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch

class ObjPushDataset(Dataset):

    def __init__(self, dirname, size=None):
        self.dirname = dirname
        self.data = np.loadtxt(dirname + '.txt')
        self.size = size

    def __len__(self):
        if self.size: # in cases where I want to define size
            return self.size
        else:
            return len(self.data)

    def __getitem__(self, idx):
        '''idx should be a single value, not list'''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx == -1:
            idx = self.__len__() - 1 # a hack to prevent accessing img1 out of range
        if idx > self.__len__()-1:
            raise ValueError('idx must be smaller than the len-1\
                                in order to get two images')

        obj1 = self.data[idx, 5:7]
        # fix the last idx using identity push (i.e. zero push)
        if idx == self.__len__()-1:
            obj2 = obj1
            push = np.zeros((4,), dtype='float32')
        else:
            obj2 = self.data[idx+1, 5:7]
            push = np.float32(self.data[idx, 1:5])

        push = np.array(push)

        sample = {'obj1': obj1, 'obj2': obj2, 'push': push}
        return sample 

train_dir = 'push_dataset/train'
test_dir = 'push_dataset/test'
bsize = 64

train_loader = DataLoader(ObjPushDataset(train_dir), batch_size=bsize, shuffle=True)
valid_loader = DataLoader(ObjPushDataset(test_dir), batch_size=bsize, shuffle=True)  