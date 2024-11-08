import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import math


class ResBlock(nn.Module):
    def __init__(self, indim, outdim, kernel_size=3):
        super(ResBlock, self).__init__()
        trunk = [nn.ReLU(inplace=True), nn.Conv2d(indim, outdim, kernel_size, 1, kernel_size//2)] + [nn.ReLU(inplace=True), nn.Conv2d(outdim, outdim, kernel_size, 1, kernel_size//2)]
        self.layers = nn.Sequential(*trunk)

    def forward(self, x):
        res = x
        x_ = x + 1 - 1
        out = self.layers(x_)
        out = out +res
        return out

class SpeAttBlock(nn.Module):
    def __init__(self, nFeat):
        super(SpeAttBlock, self).__init__()
        trunk = [nn.Conv2d(nFeat*2, nFeat, 3, 1, 1),
                ResBlock(nFeat, nFeat),
                nn.Conv2d(nFeat, nFeat, 4, 4, 0),
                ResBlock(nFeat, nFeat),
                nn.Conv2d(nFeat, nFeat, 4, 4, 0),
                nn.AdaptiveAvgPool2d(1)
                ]
        self.trunk = nn.Sequential(*trunk)
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        out = self.trunk(x)
        out = self.Sig(out)
        return out

class SpaAttBlock(nn.Module):
    def __init__(self, nFeat):
        super(SpaAttBlock, self).__init__()
        trunk = [nn.Conv2d(nFeat*2, nFeat, 3, 1, 1),
                ResBlock(nFeat, nFeat),
                ResBlock(nFeat, nFeat),
                nn.Conv2d(nFeat, nFeat, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(nFeat, nFeat, 3, 1, 1)]
        self.trunk = nn.Sequential(*trunk)
        self.Sig = nn.Sigmoid()
    def forward(self, x):
        out = self.trunk(x)
        out = self.Sig(out)
        return out

class DABlock(nn.Module):
    def __init__(self, nFeat):
        super(DABlock, self).__init__()
        self.conv = nn.Conv2d(nFeat*3, nFeat, 3, 1, 1)
        self.conv2 = nn.Conv2d(nFeat, nFeat, 3, 1, 1)
        self.relu = nn.LeakyReLU()
        self.spe_att = SpeAttBlock(nFeat)
        self.spa_att = SpaAttBlock(nFeat)

    def forward(self, x, z, y):
        spe_att_map = self.spe_att(torch.cat((z, x), 1))
        spa_att_map = self.spa_att(torch.cat((z, y), 1))
        out_x = x * spe_att_map
        out_y = y * spa_att_map
        out_z = z
        out = self.relu(self.conv(torch.cat((out_x, out_z, out_y), 1)))
        out = self.conv2(out)
        return out

class make_dilation_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3, dropout=False, dilation=1):
    super(make_dilation_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size + 2*(dilation-1) -1)//2, bias=True, dilation=dilation)
    self.dp = dropout
    self.relu = nn.ReLU(inplace=True)
  def forward(self, x):
    x_ = x + 1 - 1
    out = self.relu(self.conv(x_))
    out = torch.cat((x_, out), 1)
    return out

### Residual Wide Dense Block (RWDB)
class RWDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(RWDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dilation_dense(nChannels_, growthRate, dilation=1))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = nn.Conv2d(nChannels_+3*nChannels, nChannels, kernel_size=1, padding=0, bias=True)
    self.conv9 = nn.Conv2d(nChannels, nChannels, 9, 1, 4)
    self.conv7 = nn.Conv2d(nChannels, nChannels, 7, 1, 3)
    self.conv5 = nn.Conv2d(nChannels, nChannels, 5, 1, 2)
    self.relu = nn.LeakyReLU()
  def forward(self, x):
    out1 = self.dense_layers(x)
    out5 = self.relu(self.conv5(x))
    out7 = self.relu(self.conv7(x))
    out9 = self.relu(self.conv9(x))
    out = self.conv_1x1(torch.cat((out1, out5, out7, out9), 1))
    out = out + x
    return out



### Multistage dual-attention guided fusion network
class MDANet(nn.Module):
    def __init__(self, Dim=[1, 32, 31], depth=1, nDenselayer=4, nFeat=64, growthRate=32):
        super(MDANet, self).__init__()

        block1_1 = []
        block1_2 = []
        block1_3 = []
        block2_1 = []
        block2_2 = []
        block2_3 = []
        block3_1 = []
        block3_2 = []
        block3_3 = []

        for i in range(depth):
            block1_1.append(RWDB(nFeat, nDenselayer, growthRate))
            block1_2.append(RWDB(nFeat, nDenselayer, growthRate))
            block1_3.append(RWDB(nFeat, nDenselayer, growthRate))
            block2_1.append(RWDB(nFeat, nDenselayer, growthRate))
            block2_2.append(RWDB(nFeat, nDenselayer, growthRate))
            block2_3.append(RWDB(nFeat, nDenselayer, growthRate))
            block3_1.append(RWDB(nFeat, nDenselayer, growthRate))
            block3_2.append(RWDB(nFeat, nDenselayer, growthRate))
            block3_3.append(RWDB(nFeat, nDenselayer, growthRate))

        self.conv1_1 = nn.Sequential(*[nn.Conv2d(Dim[0], nFeat, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(nFeat, nFeat, 3, 1, 1)])
        self.conv2_1 = nn.Sequential(*[nn.Conv2d(Dim[2], nFeat, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(nFeat, nFeat, 3, 1, 1)])
        self.conv3_1 = nn.Sequential(*[nn.Conv2d(Dim[1], nFeat, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(nFeat, nFeat, 3, 1, 1)])

        self.da1 = DABlock(nFeat)
        self.da2 = DABlock(nFeat)
        self.da3 = DABlock(nFeat)
        self.da4 = DABlock(nFeat)


        self.branch1_1 = nn.Sequential(*block1_1)
        self.branch1_2 = nn.Sequential(*block1_2)
        self.branch1_3 = nn.Sequential(*block1_3)
        self.branch2_1 = nn.Sequential(*block2_1)
        self.branch2_2 = nn.Sequential(*block2_2)
        self.branch2_3 = nn.Sequential(*block2_3)
        self.branch3_1 = nn.Sequential(*block3_1)
        self.branch3_2 = nn.Sequential(*block3_2)
        self.branch3_3 = nn.Sequential(*block3_3)
        self.branch_out_1 = RWDB(nFeat, nDenselayer, growthRate)
        self.branch_out_2 = RWDB(nFeat, nDenselayer, growthRate)

        self.conv2 = nn.Conv2d(nFeat, Dim[2], 3, 1 ,1)

        self.Relu = nn.ReLU(inplace=True)
        self.Sig = nn.Sigmoid()

    def forward(self, Input):
        [x, z, y] = Input
        res = y

        ### Input stage
        out1 = self.conv1_1(x)
        out2 = self.conv2_1(y)
        out3 = self.conv3_1(z)
        res1 = out1
        res2 = out2
        res3 = out3

        out1 = self.branch1_1(out1)
        out2 = self.branch2_1(out2)
        out3 = self.branch3_1(out3)
        out3 = self.da1(out1, out3, out2)

        out1 = self.branch1_2(out1)
        out2 = self.branch2_2(out2)
        out3 = self.branch3_2(out3)
        out3 = self.da2(out1, out3, out2)

        out1 = self.branch1_3(out1)
        out2 = self.branch2_3(out2)
        out3 = self.branch3_3(out3)
        out3 = self.da3(out1, out3, out2)

        ### Reconstruction stage
        out = self.branch_out_1(out3)
        out = self.branch_out_2(out)
        out = self.conv2(out) + res

        return out
