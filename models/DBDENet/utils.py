# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

"""
ResidualBlock is used for feature extraction 
"""
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, p=1):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=k, padding=p),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return x + self.net(x)