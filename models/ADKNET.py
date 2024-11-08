# -*- encoding: utf-8 -*-
'''
@File    :   ADKNET.py
@Time    :   2023/07/03 11:45:52
@Author  :   Liuch(laxio) 
@Version :   1.0
@Source  :   Source-Adaptive Discriminative Kernels based Network for Remote Sensing Pansharpening(IJCAI2022)
@Note    :   batch_size = 32, learning_rate = 0.003, L1, 1000 epoch, decay 100, x0.5, ADAM
@Contact :   isliuch@yeah.net
'''

# here put the import lib
import torch.nn as nn
from torchsummary import summary
from modules.ADKG import ADKGenerator
#from ADKG import Fast_ADKGenerator as ADKGenerator


class ADKLayer(nn.Module):
    def __init__(self, channel=60, kernel_size=3, nonlinearity='leaky_relu',
                 stride=1, se_ratio=0.05):
        super(ADKLayer, self).__init__()
        self.ADKGenerator = ADKGenerator(in_channels=channel, kernel_size=kernel_size,
                               nonlinearity=nonlinearity, stride=stride,
                               se_ratio=se_ratio)

    def forward(self, x, y):
        x = self.ADKGenerator(x, y)
        return x


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()

        channel = 60
        spectral_num = 8

        self.deconv = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num,
                                         kernel_size=8, stride=4, padding=2, bias=True)
        self.upsample = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num,
                                           kernel_size=8, stride=4, padding=2, bias=True)

        self.adk1 = ADKLayer()
        self.adk2 = ADKLayer()
        self.adk3 = ADKLayer()
        self.adk4 = ADKLayer()
        self.adk5 = ADKLayer()
        self.adk6 = ADKLayer()
        self.adk7 = ADKLayer()
        self.relu = nn.PReLU()

        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=channel, kernel_size=5, padding=2)

    def forward(self, x, y):  # x stands for lrms, and y stands for pan

        skip = self.upsample(x)

        # forward propagation
        x = self.relu(self.deconv(x))
        x = self.relu(self.conv1(x))
        y = self.relu(self.conv3(y))

        # 7 ADK layers
        x = self.adk1(x, y)
        x = self.adk2(x, y)
        x = self.adk3(x, y)
        x = self.adk4(x, y)
        x = self.adk5(x, y)
        x = self.adk6(x, y)
        x = self.adk7(x, y)

        x = self.conv2(x)

        return x + skip


def summaries(model, grad=False):
    if grad:
        summary(model, input_size=[(8, 16, 16), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)