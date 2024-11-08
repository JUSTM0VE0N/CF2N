# -*- encoding: utf-8 -*-
'''
@File    :   SRPPNN.py
@Time    :   2023/07/03 02:04:52
@Author  :   Liuch(laxio) 
@Version :   1.0
@Source  :   Super-Resolution-Guided Progressive Pansharpening Based on a Deep Convolutional Neural Network(TGRS2021)
@Note    :   batch_size = 64, MSE, Adam, 0.0001, 1000 epoch, decay 1000, x0.1
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
        summary(model, input_size=[(8, 16, 16), (8, 64, 64), (1, 64, 64)], batch_size=1)
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

        num_channels = 8  #4
        n_resblocks = 11

        res_block_s1 = [
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s1.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s1.append(Upsampler(2, 32, activation='prelu'))
        res_block_s1.append(ConvBlock(32, 8, 3, 1, 1, activation='prelu', norm=None, bias = False))  # 4
        self.res_block_s1 = nn.Sequential(*res_block_s1)

        res_block_s2 = [
            ConvBlock(num_channels+1, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s2.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s2.append(ConvBlock(32, 8, 3, 1, 1, activation='prelu', norm=None, bias = False))  # 4
        self.res_block_s2 = nn.Sequential(*res_block_s2)
        
        res_block_s3 = [
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s3.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s3.append(Upsampler(2, 32, activation='prelu'))
        res_block_s3.append(ConvBlock(32, 8, 3, 1, 1, activation='prelu', norm=None, bias = False))  # 4
        self.res_block_s3 = nn.Sequential(*res_block_s3)

        res_block_s4 = [
            ConvBlock(num_channels+1, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s4.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        res_block_s4.append(ConvBlock(32, 8, 3, 1, 1, activation='prelu', norm=None, bias = False))  # 4
        self.res_block_s4 = nn.Sequential(*res_block_s4)

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
        self.drop_path_prob = 0.0

    def forward(self, l_ms, b_ms, x_pan):

        hp_pan_4 = x_pan - F.interpolate(F.interpolate(x_pan, scale_factor=1/4, mode='bicubic'), scale_factor=4, mode='bicubic')
        lr_pan = F.interpolate(x_pan, scale_factor=1/2, mode='bicubic')
        hp_pan_2 = lr_pan - F.interpolate(F.interpolate(lr_pan, scale_factor=1/2, mode='bicubic'), scale_factor=2, mode='bicubic')
        
        s1 = self.res_block_s1(l_ms)
        s1 = s1 + F.interpolate(l_ms, scale_factor=2, mode='bicubic')
        s2 = self.res_block_s2(torch.cat([s1, lr_pan], 1)) + F.interpolate(l_ms, scale_factor=2, mode='bicubic') + hp_pan_2
        s3 = self.res_block_s3(s2) + b_ms
        s4 = self.res_block_s4(torch.cat([s3, x_pan], 1)) + b_ms + hp_pan_4
        
        return s4