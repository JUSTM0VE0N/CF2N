# -*- encoding: utf-8 -*-
'''
@File    :   AdaConv.py
@Time    :   2023/03/03 11:41:52
@Author  :   Liuch(laxio) 
@Version :   1.0
@Source  :   own
@Note    :   own
@Contact :   isliuch@yeah.net
'''

# here put the import lib
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from modules.AttenConvmix import ACmix

class TSAM(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, padding=0, dilation=1,
                 groups=1, use_spectral_bias=True):
        super(TSAM, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_attn = kernel_att
        self.head = head
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_spectral_bias

        # Spatial adaptive weights
        self.attention1 = nn.Sequential(
            # ACmix(in_planes, kernel_conv ** 2, kernel_att, head=3, kernel_conv=3, stride=stride, dilation=dilation),
            nn.Conv2d(in_planes, kernel_conv**2, kernel_conv , stride, padding),
            nn.ReLU(inplace=False),
            #ACmix(kernel_conv ** 2, kernel_conv ** 2, kernel_att, head=3, kernel_conv=3, stride=stride, dilation=dilation),
            nn.Conv2d(kernel_conv **2, kernel_conv**2,1),
            nn.ReLU(inplace=False),
            # ACmix(kernel_conv**2, kernel_conv**2, kernel_att, head=3, kernel_conv=3, stride=stride, dilation=dilation),
            nn.Conv2d(kernel_conv**2, kernel_conv**2, 1),
            nn.Sigmoid()
        )  # b,9,H,W
        # self.attention2=nn.Sequential(
        #     nn.Conv2d(in_planes,(kernel_size**2)*in_planes,kernel_size, stride, padding,groups=in_planes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d((kernel_size**2)*in_planes,(kernel_size**2)*in_planes,1,groups=in_planes),
        #     nn.Sigmoid()
        # ) #b,9n,H,W
        if use_spectral_bias == True:  # Global spectral adaptive weights
            self.attention3 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                # ACmix(in_planes, out_planes),
                nn.Conv2d(in_planes, out_planes, 1),
                nn.ReLU(inplace=True),
                # ACmix(out_planes, out_planes, kernel_att, head, kernel_conv, stride=stride, dilation=dilation)
                nn.Conv2d(out_planes, out_planes, 1)
            )  # b,m,1,1

        conv1 = nn.Conv2d(in_planes, out_planes, kernel_conv, stride, padding, dilation, groups)
        self.weight = conv1.weight  # m, n, k, k

    def forward(self, pan):
        (b, n, H, W) = pan.shape  # (b, 1, 64, 64)
        m = self.out_planes  # 8
        k = self.kernel_conv  # 3
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)
        atw1 = self.attention1(pan)  # b,k*k,n_H,n_W                     # (b, 9, 64, 64)
        # atw2=self.attention2(x) #b,n*k*k,n_H,n_W

        atw1 = atw1.permute([0, 2, 3, 1])  # b,n_H,n_W,k*k # (b, 64, 64, 9)
        atw1 = atw1.unsqueeze(3).repeat([1, 1, 1, n, 1])  # b,n_H,n_W,n,k*k    (b, 64, 64, 1, 9)
        atw1 = atw1.view(b, n_H, n_W, n * k * k)  # b,n_H,n_W,n*k*k               (b, 64, 64, 9)

        # atw2=atw2.permute([0,2,3,1]) #b,n_H,n_W,n*k*k

        atw = atw1  # *atw2 #b,n_H,n_W,n*k*k                                (b, 64, 64, 9)
        atw = atw.view(b, n_H * n_W, n * k * k)  # b,n_H*n_W,n*k*k                 (b, 64*64, 9)
        atw = atw.permute([0, 2, 1])  # b,n*k*k,n_H*n_W                      (b, 9, 64*64)

        kx = F.unfold(pan, kernel_size=k, stride=self.stride, padding=self.padding)  # b,n*k*k,n_H*n_W   (b, 9, 64*64)
        atx = atw * kx  # b,n*k*k,n_H*n_W  (b, 9, 64*64)

        atx = atx.permute([0, 2, 1])  # b,n_H*n_W,n*k*k    (b, 64*64, 9)
        atx = atx.view(1, b * n_H * n_W, n * k * k)  # 1,b*n_H*n_W,n*k*k   (1, b*64*64, 9)

        w = self.weight.view(m, n * k * k)  # m,n*k*k   (8, 9)
        w = w.permute([1, 0])  # n*k*k,m            (9, 8)
        y = torch.matmul(atx, w)  # 1,b*n_H*n_W,m   (1, b*64*64, 8)
        y = y.view(b, n_H * n_W, m)  # b,n_H*n_W,m     (b, 64*64, 8)
        if self.bias == True:
            bias = self.attention3(pan)  # b,m,1,1
            bias = bias.view(b, m).unsqueeze(1)  # b,1,m
            bias = bias.repeat([1, n_H * n_W, 1])  # b,n_H*n_W,m
            y = y + bias  # b,n_H*n_W,m

        y = y.permute([0, 2, 1])  # b,m,n_H*n_W
        y = F.fold(y, output_size=(n_H, n_W), kernel_size=1)  # b,m,n_H,n_W   (b, 8, 64, 64)
        return y


class SICM(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, padding=0, dilation=1,
                 groups=1, use_spatial_bias=True):
        super(SICM, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_attn = kernel_att
        self.head = head
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_spatial_bias

        # Spectral adaptive weights
        self.attention1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # ACmix(in_planes, kernel_conv**2, kernel_att, head=3, kernel_conv=3, stride=stride, dilation=dilation),
            nn.Conv2d(in_planes, kernel_conv ** 2, 1),
            nn.ReLU(inplace=True),
            # ACmix(kernel_conv**2, kernel_conv**2, kernel_att, head=3, kernel_conv=3, stride=stride, dilation=dilation)
            nn.Conv2d(kernel_conv ** 2, kernel_conv ** 2, 1)
        )  # b,m,1,1
        # self.attention2=nn.Sequential(
        #     nn.Conv2d(in_planes,(kernel_size**2)*in_planes,kernel_size, stride, padding,groups=in_planes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d((kernel_size**2)*in_planes,(kernel_size**2)*in_planes,1,groups=in_planes),
        #     nn.Sigmoid()
        # ) #b,9n,H,W
        if use_spatial_bias == True:  # Global spatial adaptive weights
            self.attention3 = nn.Sequential(
                # ACmix(in_planes, kernel_conv**2, kernel_att, head=3, kernel_conv=3, stride=stride, dilation=dilation),
                nn.Conv2d(in_planes, out_planes, kernel_conv, stride, padding),
                nn.ReLU(inplace=True),
                # ACmix(kernel_conv**2, kernel_conv**2, kernel_att, head=3, kernel_conv=3, stride=stride, dilation=dilation),
                nn.Conv2d(out_planes, out_planes, 1),
                nn.ReLU(inplace=True),
                # ACmix(kernel_conv ** 2, kernel_conv ** 2, kernel_att, head=3, kernel_conv=3, stride=stride, dilation=dilation),
                nn.Conv2d(out_planes, out_planes, 1),
                nn.Sigmoid()
            )  # b,9,H,W

        conv1 = nn.Conv2d(in_planes, out_planes, kernel_conv, stride, padding, dilation, groups)
        self.weight = conv1.weight  # m, n, k, k

    def forward(self, lms):
        (b, n, H, W) = lms.shape  # (b, 8, 64, 64)
        m = self.out_planes
        k = self.kernel_conv
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)
        atw1 = self.attention1(lms)  # b,k*k,1,1                                                    # (b, 9, 1, 1)
        # atw2=self.attention2(x) #b,n*k*k,n_H,n_W

        atw1 = atw1.permute([0, 2, 3, 1])  # b,1,1,k*k                                                 # (b, 1, 1, 9)
        atw1 = atw1.unsqueeze(3).repeat(
            [1, n_H, n_W, n, 1])  # b,n_H,n_W,n,k*k                         # (b, 64, 64, 8, 9)
        atw1 = atw1.view(b, n_H, n_W,
                         n * k * k)  # b,n_H,n_W,n*k*k                                        # (b, 64, 64, 72)

        # atw2=atw2.permute([0,2,3,1]) #b,n_H,n_W,n*k*k

        atw = atw1  # *atw2 #b,n_H,n_W,n*k*k
        atw = atw.view(b, n_H * n_W,
                       n * k * k)  # b,n_H*n_W,n*k*k                                          # (b, 64*64, 72)
        atw = atw.permute([0, 2, 1])  # b,n*k*k,n_H*n_W                                               # (b, 72, 64*64)

        kx = F.unfold(lms, kernel_size=k, stride=self.stride, padding=self.padding)  # b,n*k*k,n_H*n_W # (b, 72, 64*64)
        atx = atw * kx  # b,n*k*k,n_H*n_W                                                             # (b, 72, 64*64)

        atx = atx.permute([0, 2, 1])  # b,n_H*n_W,n*k*k                                               # (b, 64*64, 72)
        atx = atx.view(1, b * n_H * n_W,
                       n * k * k)  # 1,b*n_H*n_W,n*k*k                                      # (1, b*64*64, 72)

        w = self.weight.view(m, n * k * k)  # m,n*k*k                                                    # (8, 72)
        w = w.permute([1, 0])  # n*k*k,m                                                             # (72, 8)
        y = torch.matmul(atx, w)  # 1,b*n_H*n_W,m                                                    # (1, b*64*64, 8)
        y = y.view(b, n_H * n_W, m)  # b,n_H*n_W,m                                                      # (b, 64*64, 8)

        if self.bias == True:
            bias = self.attention3(lms)  # b,m,n_H,n_W                                              # (b, 8, 64, 64)
            biass = bias.permute([0, 2, 3, 1])  # b. n_H, n_W, m                                # (b, 64, 64, 8)
            biass = biass.view(b, n_H * n_W, m)  # (b, 64*64, 8)
            # bias=bias.view(b,m).unsqueeze(1) #b,1,m
            # bias=bias.repeat([1,n_H*n_W,1]) #b,n_H*n_W,m
            y = y + biass  # b,n_H*n_W,m

        y = y.permute([0, 2, 1])  # b,m,n_H*n_W                                                       # (b, 8, 64*64)
        y = F.fold(y, output_size=(n_H, n_W), kernel_size=1)  # b,m,n_H,n_W                            # (b, 8, 64, 64)
        return y


class AFM(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, padding=0, dilation=1,
                 groups=1, use_bias=True):
        super(AFM, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_attn = kernel_att
        self.head = head
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_bias

        # Generating local adaptive weights
        self.attention1 = nn.Sequential(
            ACmix(in_planes, kernel_conv * kernel_conv, kernel_att, head=3, kernel_conv=3, stride=stride,
                  dilation=dilation),
            # nn.Conv2d(in_planes, kernel_size**2, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            ACmix(kernel_conv ** 2, kernel_conv ** 2, kernel_att, head=3, kernel_conv=3, stride=stride,
                  dilation=dilation),
            # nn.Conv2d(kernel_size**2,kernel_size**2,1),
            nn.ReLU(inplace=True),
            ACmix(kernel_conv ** 2, kernel_conv ** 2, kernel_att, head=3, kernel_conv=3, stride=stride,
                  dilation=dilation),
            # nn.Conv2d(kernel_size ** 2, kernel_size ** 2, 1),
            nn.Sigmoid()
        )  # b,9,H,W
        # self.attention2=nn.Sequential(
        #     nn.Conv2d(in_planes,(kernel_size**2)*in_planes,kernel_size, stride, padding,groups=in_planes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d((kernel_size**2)*in_planes,(kernel_size**2)*in_planes,1,groups=in_planes),
        #     nn.Sigmoid()
        # ) #b,9n,H,W
        self.attention2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # ACmix(in_planes, out_planes),
            # nn.Conv2d(in_planes, out_planes, 1),
            nn.Conv2d(in_planes, kernel_conv ** 2, 1),
            nn.ReLU(inplace=True),
            # ACmix(out_planes, out_planes, kernel_att, head, kernel_conv, stride=stride, dilation=dilation)
            # nn.Conv2d(out_planes, out_planes, 1)
            nn.Conv2d(kernel_conv ** 2, kernel_conv ** 2, 1)
        )
        if use_bias == True:  # Global local adaptive weights
            self.attention3 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                # ACmix(in_planes, out_planes),
                nn.Conv2d(in_planes, out_planes, 1),
                # nn.Conv2d(in_planes, kernel_conv**2, 1),
                nn.ReLU(inplace=True),
                # ACmix(out_planes, out_planes, kernel_att, head, kernel_conv, stride=stride, dilation=dilation)
                nn.Conv2d(out_planes, out_planes, 1)
                # nn.Conv2d(kernel_conv**2, kernel_conv**2, 1)
            )  # b,k*k,1,1

        conv1 = nn.Conv2d(in_planes, out_planes, kernel_conv, stride, padding, dilation, groups)
        self.weight = conv1.weight  # m, n, k, k

    def forward(self, x):
        (b, n, H, W) = x.shape
        m = self.out_planes
        k = self.kernel_conv
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)
        atw1 = self.attention1(x)  # b,k*k,n_H,n_W
        # atw2=self.attention2(x) #b,n*k*k,n_H,n_W

        atw1 = atw1.permute([0, 2, 3, 1])  # b,n_H,n_W,k*k
        atw1 = atw1.unsqueeze(3).repeat([1, 1, 1, n, 1])  # b,n_H,n_W,n,k*k
        atw1 = atw1.view(b, n_H, n_W, n * k * k)  # b,n_H,n_W,n*k*k

        # atw2=atw2.permute([0,2,3,1]) #b,n_H,n_W,n*k*k

        atw = atw1  # *atw2 #b,n_H,n_W,n*k*k
        atw = atw.view(b, n_H * n_W, n * k * k)  # b,n_H*n_W,n*k*k
        atw = atw.permute([0, 2, 1])  # b,n*k*k,n_H*n_W
        # if self.bias == True:
        ratio = self.attention2(x)  # (b, k*k, 1, 1)
        ratio = ratio.permute([0, 2, 3, 1])  # (b, 1, 1, k*k)
        ratio = ratio.unsqueeze(3).repeat([1, n_H, n_W, n, 1])
        ratio = ratio.view(b, n_H*n_W, n * k * k)
        ratio = ratio.permute([0, 2, 1])
        atw = torch.mul(atw, ratio)
        kx = F.unfold(x, kernel_size=k, stride=self.stride, padding=self.padding)  # b,n*k*k,n_H*n_W
        atx = atw * kx  # b,n*k*k,n_H*n_W

        atx = atx.permute([0, 2, 1])  # b,n_H*n_W,n*k*k
        atx = atx.view(1, b * n_H * n_W, n * k * k)  # 1,b*n_H*n_W,n*k*k

        w = self.weight.view(m, n * k * k)  # m,n*k*k
        w = w.permute([1, 0])  # n*k*k,m
        y = torch.matmul(atx, w)  # 1,b*n_H*n_W,m
        y = y.view(b, n_H * n_W, m)  # b,n_H*n_W,m
        if self.bias == True:
            bias = self.attention3(x)  # b,m,1,1
            bias = bias.view(b, m).unsqueeze(1)  # b,1,m
            bias = bias.repeat([1, n_H * n_W, 1])  # b,n_H*n_W,m
            y = y + bias  # b,n_H*n_W,m   # y = torch.mul(y, bias)

        y = y.permute([0, 2, 1])  # b,m,n_H*n_W
        y = F.fold(y, output_size=(n_H, n_W), kernel_size=1)  # b,m,n_H,n_W
        return y


# SpaA_ResBlocks
class SpaARB(nn.Module):
    def __init__(self, in_planes):
        super(SpaARB, self).__init__()
        self.conv1 = TSAM(in_planes, in_planes, kernel_att=7, kernel_conv=3, stride=1, padding=1,
                                use_spectral_bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = TSAM(in_planes, in_planes, kernel_att=7, kernel_conv=3, stride=1, padding=1,
                                use_spectral_bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu1(res)
        res = self.conv2(res)
        x = x + res
        return x


# SpeA_ResBlocks
class SpeARB(nn.Module):
    def __init__(self, in_planes):
        super(SpeARB, self).__init__()
        self.conv1 = SICM(in_planes, in_planes, kernel_att=7, kernel_conv=3, stride=1, padding=1,
                                use_spatial_bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = SICM(in_planes, in_planes, kernel_att=7, kernel_conv=3, stride=1, padding=1,
                                use_spatial_bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu1(res)
        res = self.conv2(res)
        x = x + res
        return x


# LAC_ResBlocks
class AFCRB(nn.Module):
    def __init__(self, in_planes):
        super(AFCRB, self).__init__()
        self.conv1 = AFM(in_planes, in_planes, kernel_att=7, kernel_conv=3, stride=1, padding=1, use_bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = AFM(in_planes, in_planes, kernel_att=7, kernel_conv=3, stride=1, padding=1, use_bias=False)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu1(res)
        res = self.conv2(res)
        x = x + res
        return x


# Proposed Network
# class LACNET(nn.Module):
#     def __init__(self):
#         super(LACNET, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding=2)
#         # self.conv2 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2)
#         self.relu = nn.PReLU()
#
#         self.SpaARB = SpaARB(8)
#         self.SpeARB = SpeARB(8)
#
#         self.head_conv = nn.Sequential(
#             AFConv2D(16, 32, kernel_att=7, kernel_conv=3, stride=1, padding=1, use_bias=True),
#             nn.ReLU(inplace=True)
#         )
#
#         self.RB1 = AFCRB(32)
#         self.RB2 = AFCRB(32)
#         self.RB3 = AFCRB(32)
#         self.RB4 = AFCRB(32)
#         self.RB5 = AFCRB(32)
#
#         self.tail_conv = AFConv2D(32, 8, kernel_att=7, kernel_conv=3, stride=1, padding=1, use_bias=True)
#
#     def forward(self, pan, lms):
#         pan_fea = self.relu(self.conv3(pan))
#         lms_fea = self.relu(self.conv1(lms))
#         pan_spa = self.SpaARB(pan_fea)
#         lms_spe = self.SpeARB(lms_fea)
#
#         x = torch.cat([pan_spa, lms_spe], 1)
#         x = self.head_conv(x)
#         x = self.RB1(x)
#         x = self.RB2(x)
#         x = self.RB3(x)
#         x = self.RB4(x)
#         x = self.RB5(x)
#         x = self.tail_conv(x)
#         sr = lms + x
#         return sr



from torchsummary import summary

def summaries(model, grad=False):
    if grad:
        summary(model, input_size=[(1, 64, 64), (8, 64, 64)], batch_size=1, device='cpu')
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

# if __name__ == '__main__':
    # model = LACNET()
    # summaries(model, grad=True)

