import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import calculate_gain


class ChannelAtt(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpaCNN(nn.Module):
    def __init__(self, in_channels, SK_size=3, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=SK_size, padding=int(SK_size/2), stride=strides)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=SK_size, padding=int(SK_size/2), stride=strides)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        return F.relu(Y)

class SpatialCNNAtt(nn.Module):
    def __init__(self, in_channels=64, SK_size=3, kernel_size=3):
        super(SpatialCNNAtt, self).__init__()
        self.scnn = SpaCNN(in_channels=in_channels, SK_size=SK_size)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.scnn(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)