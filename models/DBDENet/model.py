# -*- coding: utf-8 -*-
"""

"""

import torch
import torch.nn as nn
from scipy.io import savemat
from utils import ResidualBlock

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, out_channels=128, scale_factor=2, k=3, p=1):
        super(UpsampleBLock, self).__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="bicubic"),
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.net(x)

"""
Funcution Name: PAN_extract
Initial param: n_residual represents the number of residualblocks
Input:  x: PAN image with shape of H * W
Out: out1: features of the middle with the shape of H * W * 128
     out2: features of the high with the shape of (H // 2) * (W // 2) * 128
"""

class PAN_extract(nn.Module):
    def __init__(self, n_residual=6):
        super(PAN_extract, self).__init__()
        self.n_residual = n_residual
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        for i in range(self.n_residual):
            self.add_module('residual1' + str(i + 1), ResidualBlock(64, 64))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        for i in range(self.n_residual):
            self.add_module('residual2' + str(i + 1), ResidualBlock(128, 128))
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

    def forward(self, x):
        # Extraction of features
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        for i in range(self.n_residual):
            out = self.__getattr__("residual1" + str(i + 1))(out)
        out1 = self.relu(self.bn3(self.conv3(out)))
        # Downsampling the features by averagepooling operation
        out = self.avgpool(out1)
        out = self.relu(self.bn4(self.conv4(out)))
        out = self.relu(self.bn5(self.conv5(out)))
        for i in range(self.n_residual):
            out = self.__getattr__("residual2" + str(i + 1))(out)
        out = self.relu(self.bn6(self.conv6(out)))
        #return the out1 and out from the PAN branch
        return out1, out

""""
Attention Mechanism
Spatial Attention is the basis of cross attention.
Init Param: Kerner_size: the receptive field of feature extraction default: 3*3 or 7*7
Input: the input feature of attention mechanism with shape of H * W * C
Output: the mask of attention mechanism with shape of H * W * 1
"""
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


"""
Function: Building the proposed network
Input:  x: PAN image with shape of H * W
        y: I image with shape of (H //4) * (W // 4)
Out: out: features of the reconstruction with shape of H * W * C
"""
class Spatial_extract(nn.Module):
    def __init__(self, n_residual=6):
        super(Spatial_extract, self).__init__()
        self.PAN_extract = PAN_extract()
        self.n_residual = n_residual
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.Attention1_1 = SpatialAttention()
        self.Attention1_2 = SpatialAttention()
        self.conv4 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(256, 102, kernel_size=1, stride=1)
        self.bn9 = nn.BatchNorm2d(102)
        self.conv10 = nn.Conv2d(256, 102, kernel_size=1, stride=1)
        self.bn10 = nn.BatchNorm2d(102)
        self.Attention2_1 = SpatialAttention()
        self.Attention2_2 = SpatialAttention()
        for i in range(self.n_residual):
            self.add_module('residual3' + str(i + 1), ResidualBlock(64, 64))
        for i in range(self.n_residual):
            self.add_module('residual4' + str(i + 1), ResidualBlock(128, 128))

    def forward(self, x, y):
        PAN_out1, PAN_out = self.PAN_extract(x)
        out = self.relu(self.bn1(self.conv1(y)))
        out = self.relu(self.bn2(self.conv2(out)))
        for i in range(self.n_residual):
            out = self.__getattr__("residual3" + str(i + 1))(out)
        out_1 = self.relu(self.bn3(self.deconv1(out)))
        PAN_out_first = torch.cat([PAN_out, out_1 * self.Attention1_1(PAN_out)], dim=1)
        PAN_out_first = self.conv4(PAN_out_first)
        out = torch.cat([out_1, PAN_out * self.Attention1_2(out_1)], dim=1)
        out = self.conv5(out)
        out_first = out + PAN_out_first
        out = self.relu(self.bn6(self.conv6(out_first)))
        out = self.relu(self.bn7(self.conv7(out)))
        for i in range(self.n_residual):
            out = self.__getattr__("residual4" + str(i + 1))(out)
        out_2 = self.relu(self.bn8(self.deconv2(out)))
        PAN_out_second = torch.cat([PAN_out1, out_2 * self.Attention2_1(PAN_out1)], dim=1)
        PAN_out_second = self.conv9(PAN_out_second)
        out = torch.cat([out_2, PAN_out1 * self.Attention2_2(out_2)], dim=1)
        out = self.conv10(out)
        out = out + PAN_out_second
        return out