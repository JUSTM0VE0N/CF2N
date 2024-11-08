# -*- encoding: utf-8 -*-
'''
@File    :   GPPNN.py
@Time    :   2023/07/03 01:38:02
@Author  :   Liuch(laxio) 
@Version :   1.0
@Source  :   Deep Gradient Projection Networks for Pan-sharpening (CVPR2021)
@Note    :   batch_size = 16, learning_rate = 1e-4, L1, 1000 epoch, decay 500, x0.5, ADAM
@Contact :   isliuch@yeah.net
'''

# here put the import lib
import os
import torch
import torch.nn as nn
import torch.optim as optim
from modules.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F

def summaries(model, writer=None, grad=False, torchsummary=None):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(4, 16, 16), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    if writer is not None:
        x = torch.randn(1, 64, 64, 64)
        writer.add_graph(model, (x,))

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()

        base_filter = 64
        ms_channel = 4
        num_channels = ms_channel+1
        n_layer = 8

        self.lr_blocks = nn.ModuleList([LRBlock(ms_channel , base_filter) for i in range(n_layer)])
        self.pan_blocks = nn.ModuleList([PANBlock(ms_channel , 1, base_filter, 1) for i in range(n_layer)])

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, l_ms, x_pan):

        _, _, m, n = l_ms.shape
        _, _, M, N = x_pan.shape
        HR = upsample(l_ms, M, N)
        
        for i in range(len(self.lr_blocks)):
            HR = self.lr_blocks[i](HR, l_ms)
            HR = self.pan_blocks[i](HR, x_pan)
            
        return HR


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels,out_channels, 3, padding=1, bias=False)
        self.relu  = nn.ReLU(True)
        
    def forward(self, x):
        x = x+self.conv2(self.relu(self.conv1(x)))
        return x
    
class BasicUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3):
        super(BasicUnit, self).__init__()
        p = kernel_size//2
        self.basic_unit = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, padding=p, bias=False),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, out_channels, kernel_size, padding=p, bias=False)
            )

    def forward(self, input):
        return self.basic_unit(input)

# LRBlock is called MSBlock in our paper
class LRBlock(nn.Module):
    def __init__(self,
                 ms_channels,
                 n_feat):
        super(LRBlock, self).__init__()
        self.get_LR = BasicUnit(ms_channels, n_feat, ms_channels)
        self.get_HR_residual = BasicUnit(ms_channels, n_feat, ms_channels)
        self.prox = BasicUnit(ms_channels, n_feat, ms_channels)
        
    def forward(self, HR, LR):
        _,_,M,N = HR.shape
        _,_,m,n = LR.shape
        
        LR_hat = upsample(self.get_LR(HR), m, n)
        LR_Residual = LR - LR_hat
        HR_Residual = upsample(self.get_HR_residual(LR_Residual), M, N)
        HR = self.prox(HR + HR_Residual)
        return HR
        
class PANBlock(nn.Module):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feat, 
                 kernel_size):
        super(PANBlock, self).__init__()
        self.get_PAN = BasicUnit(ms_channels, n_feat, pan_channels, kernel_size)
        self.get_HR_residual = BasicUnit(pan_channels, n_feat, ms_channels, kernel_size)
        self.prox = BasicUnit(ms_channels, n_feat, ms_channels, kernel_size)
        
    def forward(self, HR, PAN):
        PAN_hat = self.get_PAN(HR)
        PAN_Residual = PAN - PAN_hat
        HR_Residual = self.get_HR_residual(PAN_Residual)
        HR = self.prox(HR + HR_Residual)
        return HR
    
def upsample(x, h, w):
    return F.interpolate(x, size=[h,w], mode='bicubic', align_corners=True)