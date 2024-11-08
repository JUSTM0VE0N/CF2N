# -*- encoding: utf-8 -*-
'''
@File    :   PNN.py
@Time    :   2023/07/03 02:02:43
@Author  :   Liuch(laxio) 
@Version :   1.0
@Source  :   Pansharpening by Convolutional Neural Networks(Remote Sensing2016)
@Note    :   batch_size = 32, learning_rate = 1e-4/1e-5, L2, 5000 epoch, SGD, momentum = 0.9
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
        summary(model, input_size=[(8, 64, 64), (1, 64, 64)], batch_size=1)
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
        num_channels = 11  # 7
        out_channels = 8  # 4
        self.head = ConvBlock(num_channels, 48, 9, 1, 4, activation='relu', norm=None, bias=True)

        self.body = ConvBlock(48, 32, 5, 1, 2, activation='relu', norm=None, bias=True)

        self.output_conv = ConvBlock(32, out_channels, 5, 1, 2, activation='relu', norm=None, bias=True)

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

        NDWI = ((l_ms[:, 1, :, :] - l_ms[:, 3, :, :]) / (l_ms[:, 1, :, :] + l_ms[:, 3, :, :] + 0.0000000001)).unsqueeze(1)
        # NDWI = F.interpolate(NDWI, scale_factor=4, mode='bicubic')
        NDVI = ((l_ms[:, 3, :, :] - l_ms[:, 2, :, :]) / (l_ms[:, 3, :, :] + l_ms[:, 2, :, :]+ 0.0000000001)).unsqueeze(1)
        # NDVI = F.interpolate(NDVI, scale_factor=4, mode='bicubic')
        x_f = torch.cat([l_ms, x_pan, NDVI, NDWI], 1)
        x_f = self.head(x_f)
        x_f = self.body(x_f)
        x_f = self.output_conv(x_f)

        return x_f
    
if __name__ == '__main__':
    summaries(Net(), grad=True)