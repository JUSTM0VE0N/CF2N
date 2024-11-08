# -*- encoding: utf-8 -*-
'''
@File    :   data.py
@Time    :   2023/03/03 11:42:38
@Author  :   Liuch(laxio) 
@Version :   1.0
@Source  :   h5py
@Note    :   load *.h5 file
@Contact :   isliuch@yeah.net
'''

# here put the import lib
import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np

def get_edge(data):  # for training
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs

class Load_Trainset(data.Dataset):
    def __init__(self, file_path):
        super(Load_Trainset, self).__init__()
        dataset = h5py.File(file_path, mode='r')

        self.gt = dataset.get("gt")  # NxCxHxW
        self.ms = dataset.get("ms")
        self.lms = dataset.get("lms")
        self.pan = dataset.get("pan")

    def __getitem__(self, index):
        gt = torch.from_numpy(self.gt[index, :, :, :] / 2047).float()
        lms = torch.from_numpy(self.lms[index, :, :, :] / 2047).float()
        ms = torch.from_numpy(self.ms[index, :, :, :] / 2047).float()
        pan = torch.from_numpy(self.pan[index, :, :, :] / 2047).float()
        return gt, lms, ms, pan

    def __len__(self):
        return self.gt.shape[0]

# load reduced-resolution testing set
class Load_RRTset(data.Dataset):
    def __init__(self, file_path):
        super(Load_RRTset, self).__init__()
        dataset = h5py.File(file_path, mode='r')

        self.gt = dataset.get("gt")  # NxCxHxW
        self.ms = dataset.get("ms")
        self.lms = dataset.get("lms")
        self.pan = dataset.get("pan")

    def __getitem__(self, index):
        gt = torch.from_numpy(self.gt[index, :, :, :] / 2047).float()
        lms = torch.from_numpy(self.lms[index, :, :, :] / 2047).float()
        ms = torch.from_numpy(self.ms[index, :, :, :] / 2047).float()
        pan = torch.from_numpy(self.pan[index, :, :, :] / 2047).float()
        return gt, lms, ms, pan

    def __len__(self):
        return self.gt.shape[0]

# load full-resolution testing set
class Load_FRTset(data.Dataset):
    def __init__(self, file_path):
        super(Load_FRTset, self).__init__()
        dataset = h5py.File(file_path, mode='r')

        self.ms = dataset.get("ms")
        self.lms = dataset.get("lms")
        self.pan = dataset.get("pan")

    def __getitem__(self, index):
        lms = torch.from_numpy(self.lms[index, :, :, :] / 2047).float()
        ms = torch.from_numpy(self.ms[index, :, :, :] / 2047).float()
        pan = torch.from_numpy(self.pan[index, :, :, :] / 2047).float()
        return lms, ms, pan

    def __len__(self):
        return self.ms.shape[0]