import torch
import numpy as np
import torch.utils.data as data
import scipy.io as sio
# from scipy.misc import imresize
import copy
from Spa_downs import *



class LoadDataset_Mem(data.Dataset):
    def __init__(self, allData, patch_size=128, stride=64):
        super(LoadDataset_Mem, self).__init__()


        self.data = allData                             #The list of all data
        self.Image_size = allData[-1].shape[2]           #The size of original HR HSI
        self.P_S = patch_size                           #We devide the HR HSI into patches at first and this indicate the size of patch
        self.stride = stride                            #The stride of each patch.
        self.P_N = int(self.Image_size/self.stride)     #The number of patches.


    def __getitem__(self, Index):

        P_S = self.P_S
        S = self.stride
        P_N = self.P_N


        Image_size = self.Image_size
        Patches = P_N**2
        Image_I = int(Index/Patches)
        Patch_I = int(Index%Patches)

        HSI = self.data[Image_I]


        X = int(Patch_I/P_N) #X,Y is patch index in image
        Y = int(Patch_I%P_N)

        s = int(S/8)       ### set the scal factor as 8
        p_s = int(P_S/8)

        if X*S+P_S > Image_size and Y*S+P_S <= Image_size:
            GT = HSI[:, -P_S:, Y * S: Y * S + P_S]
            # INIT = GFF_HSI[:, -P_S:, Y * S: Y * S + P_S]
        elif X*S+P_S <= Image_size and Y*S+P_S > Image_size:
            GT = HSI[:, X * S:X * S + P_S, -P_S:]
            # INIT = GFF_HSI[:, X * S:X * S + P_S, -P_S:]
        elif X*S+P_S > Image_size and Y*S+P_S > Image_size:
            GT = HSI[:, -P_S: , -P_S: ]
            # INIT = GFF_HSI[:, -P_S: , -P_S: ]
        else:
            GT = HSI[:, X * S:X * S + P_S, Y * S:Y * S + P_S]
            # INIT = GFF_HSI[:, X * S:X * S + P_S, Y * S:Y * S + P_S]


        GT = torch.from_numpy(GT)
        # INIT = torch.from_numpy(INIT)

        return GT

    def __len__(self):

        return int(self.P_N**2*len(self.data))



class LoadDataset_Mem_Val(data.Dataset):
    def __init__(self, allValData):
        super(LoadDataset_Mem_Val, self).__init__()
        self.data = allValData                             #The list of all data

    def __getitem__(self, Index):

        HSI = self.data[Index]
        return torch.from_numpy(HSI)

    def __len__(self):

        return len(self.data)


