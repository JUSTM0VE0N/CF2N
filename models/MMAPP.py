# -*- encoding: utf-8 -*-
'''
@File    :   AdaConv.py
@Time    :   2023/03/03 15:34:31
@Author  :   Liuch(laxio)
@Version :   1.0
@Source  :   own
@Note    :   own
@Contact :   isliuch@yeah.net
'''
import torch
import torch.nn as nn
import math
import numpy as np
from modules.AdaConv import *

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

class _Conv_Block(nn.Module):
    def __init__(self):
        super(_Conv_Block, self).__init__()

        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.PixelShuffle(upscale)
            # nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        output = self.cov_block(x)
        return output

class _UpConv_Block(nn.Module):
    def __init__(self):
        super(_UpConv_Block, self).__init__()

        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
        )

class TSAResBlock(nn.Module):
    def __init__(self) -> None:
        super(TSAResBlock, self).__init__()
        self.SpaRB1 = SpaARB(32)
    
    def forward(self, x):
        x = self.SpaRB1(x)

        return x
    
class SICResBlock(nn.Module):
    def __init__(self) -> None:
        super(SICResBlock, self).__init__()
        self.SpeRB1 = SpeARB(32)
    
    def forward(self, x):
        x = self.SpeRB1(x)

        return x

class FACResBlock(nn.Module):
    def __init__(self) -> None:
        super(FACResBlock, self).__init__()
        self.RB1 = AFCRB(32)
        self.RB2 = AFCRB(32)
        self.RB3 = AFCRB(32)
        self.RB4 = AFCRB(32)
        self.RB5 = AFCRB(32)
    
    def forward(self, x):
        x = self.RB1(x)
        x = self.RB2(x)
        x = self.RB3(x)
        x = self.RB4(x)
        x = self.RB5(x)

        return x


class FACResBlock48(nn.Module):
    def __init__(self) -> None:
        super(FACResBlock48, self).__init__()
        self.RB1 = AFCRB(48)
        self.RB2 = AFCRB(48)
        self.RB3 = AFCRB(48)
        self.RB4 = AFCRB(48)
        self.RB5 = AFCRB(48)

    def forward(self, x):
        x = self.RB1(x)
        x = self.RB2(x)
        x = self.RB3(x)
        x = self.RB4(x)
        x = self.RB5(x)

        return x

class PS(nn.Module):

    def __init__(self, in_channel, up):
        super().__init__()
        self.conv_up1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * up * up, kernel_size=3,
                                 stride=1, padding=1, bias=True)
        self.up_size = up

        self.upsample = nn.PixelShuffle(self.up_size)

    def mapping(self, x):
        B, C, H, W = x.shape
        C1, H1, W1 = C // (self.up_size * self.up_size), H * self.up_size, W * self.up_size
        x = x.reshape(B, C1, self.up_size, self.up_size, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C1, H1, W1)
        return x

    def forward(self, x):
        x = self.conv_up1(x)
        return self.mapping(x)

class panbranch(nn.Module):
    def __init__(self) -> None:
        super(panbranch, self).__init__()
        self.head_conv = TSAM(in_planes=1, out_planes=32, kernel_att=7, kernel_conv=3, stride=1, padding=1,use_spectral_bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.E1 = self.makelayers(TSAResBlock)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relue1 = nn.ReLU(inplace=True)
        self.conv_d2_out = TSAM(in_planes=32, out_planes=8, kernel_att=7, kernel_conv=3, stride=1, padding=1, use_spectral_bias=True)

        self.mid_conv = TSAM(in_planes=32, out_planes=32, kernel_att=7, kernel_conv=3, stride=1, padding=1, use_spectral_bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.E2 = self.makelayers(TSAResBlock)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relue2 = nn.ReLU(inplace=True)
        self.conv_d4_out = TSAM(in_planes=32, out_planes=8, kernel_att=7, kernel_conv=3, stride=1, padding=1, use_spectral_bias=True)

        init_weights(self.head_conv, self.E1, self.mp1, self.conv_d2_out, self.mid_conv, self.E2, self.mp2,
                     self.conv_d4_out)

    def makelayers(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        pan = self.head_conv(x)
        pan = self.relu1(pan)
        pan_e1 = self.E1(pan) + pan
        pan_e1 = self.relue1(pan_e1)
        pan_d2 = self.mp1(pan_e1)
        pan_d2_out = self.conv_d2_out(pan_d2)

        pan_mid = self.mid_conv(pan_d2)
        pan_mid = self.relu2(pan_mid)
        pan_e2 = self.E2(pan_mid) + pan_d2
        pan_e2 = self.relue2(pan_e2)
        pan_d4 = self.mp2(pan_e2)
        pan_d4_out = self.conv_d4_out(pan_d4)

        return x, pan_d2_out, pan_d4_out

class msbranch(nn.Module):
    def __init__(self) -> None:
        super(msbranch, self).__init__()
        self.head_conv = SICM(in_planes=8, out_planes=32, kernel_att=7, kernel_conv=3, stride=1, padding=1,
                   use_spatial_bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.E1 = self.makelayers(SICResBlock)
        self.up1 = PS(32, 2)
        self.relue1 = nn.ReLU(inplace=True)
        self.conv_u2_out = SICM(in_planes=32, out_planes=8, kernel_att=7, kernel_conv=3, stride=1, padding=1,
                   use_spatial_bias=True)
        self.out_relu = nn.ReLU(inplace=True)

        self.mid_conv = SICM(in_planes=32, out_planes=32, kernel_att=7, kernel_conv=3, stride=1, padding=1,
                                      use_spatial_bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.E2 = self.makelayers(SICResBlock)
        self.up2 = PS(32, 2)
        self.relue2 = nn.ReLU(inplace=True)
        self.conv_u4_out = SICM(in_planes=32, out_planes=8, kernel_att=7, kernel_conv=3, stride=1, padding=1,
                                      use_spatial_bias=True)
        init_weights(self.head_conv, self.E1, self.up1, self.conv_u2_out, self.mid_conv, self.E2, self.up2,
                     self.conv_u4_out)

    def makelayers(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        ms = self.head_conv(x)
        ms = self.relu1(ms)
        ms_e1 = self.E1(ms) + ms
        ms_e1 = self.relue1(ms_e1)
        ms_u2 = self.up1(ms_e1)
        ms_u2 = self.out_relu(ms_u2)
        ms_u2_out = self.conv_u2_out(ms_u2)

        ms_mid = self.mid_conv(ms_u2)
        ms_mid = self.relu2(ms_mid)
        ms_e2 = self.E2(ms_mid) + ms_u2
        ms_e2 = self.relue2(ms_e2)
        ms_u4 = self.up2(ms_e2)
        ms_u4 = self.out_relu(ms_u4)
        ms_u4_out = self.conv_u4_out(ms_u4)

        return x, ms_u2_out, ms_u4_out


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.head_conv = AFM(16, 32, kernel_att=7, kernel_conv=3, stride=1, padding=1, use_bias=False)
        self.head_relu = nn.PReLU()
        self.pb = panbranch()
        self.mb = msbranch()

        self.F1 = self.makelayers(FACResBlock)
        self.conv_f1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1, bias=True)  # out1

        self.up1 = PS(32, 2)
        self.up1_relu = nn.LeakyReLU(0.2, inplace=True)  # I1

        self.F2 = self.makelayers(FACResBlock48)
        self.conv_f2 = nn.Conv2d(in_channels=48, out_channels=8, kernel_size=3, stride=1, padding=1, bias=True)  # out2

        self.up2 = PS(48, 2)
        self.up2_relu = nn.LeakyReLU(0.2, inplace=True)  # I2

        self.F3 = self.makelayers(FACResBlock)
        self.conv_f3 = nn.Conv2d(in_channels=48, out_channels=8, kernel_size=3, stride=1, padding=1, bias=True)  # out2

        self.tail_conv = nn.Sequential(
            AFM(17, 8, kernel_att=7, kernel_conv=3, stride=1, padding=1, use_bias=False),
            nn.ReLU(inplace=True),
            AFM(8, 8, kernel_att=7, kernel_conv=3, stride=1, padding=1, use_bias=False),
        )

        init_weights(self.head_conv, self.pb, self.mb, self.F1, self.conv_f1, self.up1, self.F2, self.conv_f2,
                     self.up2, self.F3, self.conv_f3, self.tail_conv)

    def makelayers(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x, y):  # x: ms    y:pan
        pan, pan_d2, pan_d4 = self.pb(y)
        ms, ms_u2, ms_u4 = self.mb(x)
        f1 = torch.cat((ms, pan_d4), 1)
        f1 = self.head_relu(self.head_conv(f1))
        f1 = self.F1(f1)    # 32
        out1 = self.conv_f1(f1)    # (12, 8, 16, 16)

        f2_pre = self.up1_relu(self.up1(f1))   # (12, 32, 32, 32)
        f2 = torch.cat((f2_pre, ms_u2, pan_d2), 1)  # (12, 48, 32, 32)
        f2 = self.F2(f2)  # (12, 48, 32, 32)
        out2 = self.conv_f2(f2)   # (12, 8, 32, 32)

        f3_pre = self.up2_relu(self.up2(f2))
        f3_pre = self.conv_f3(f3_pre)  # this f3
        f3 = torch.cat((f3_pre, pan, ms_u4), 1)
        out3 = self.tail_conv(f3)

        return out1, out2, out3, ms_u4

def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                variance_scaling_initializer(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x / 10 * 1.28

    variance_scaling(tensor)

    return tensor

from torchsummary import summary
def summaries(model, grad=False):
    if grad:
        summary(model, input_size=[(8, 16, 16), (1, 64, 64)], batch_size=1, device='cpu')
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)


if __name__ == '__main__':
    import time
    model = Network().cuda()
    pan = torch.rand(1, 1, 64, 64).cuda()
    lms = torch.rand(1, 8, 64, 64).cuda()
    ms = torch.rand(1, 8, 16, 16).cuda()
    t = time.time()
    _, _, _, T = model(ms, pan)
    tt = time.time() - t
    print(tt)
    from thop import profile
    macs, params = profile(model, (ms, pan))
    print('Computational complexity: ', str(macs / 1000 ** 2) + 'M', 'Number of parameters: ', str(params / 1000) + 'K')

