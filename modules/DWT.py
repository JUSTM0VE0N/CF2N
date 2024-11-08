#coding:utf-8
import torch.nn as nn
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    # print('---------------- LL ------------------')
    # print(x_LL[:,0,:,:])
    # print()
    # print('---------------- HL ------------------')
    # print(x_HL[:,0,:,:])
    # print()
    # print('---------------- LH ------------------')
    # print(x_LH[:,0,:,:])
    # print()
    # print('---------------- HH ------------------')
    # print(x_HH[:,0,:,:])
    return x_LL, x_HL, x_LH, x_HH


# 使用哈尔 haar 小波变换来实现二维逆向离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    # print('-------------- enter iwt ---------------')
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    # print('-------------- back ---------------')
    return h


# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    from torchvision import transforms as trans
    import matplotlib.pyplot as plt

    img = Image.open('./lakers.jpg')
    transform = trans.Compose([
        trans.ToTensor()
    ])
    img = transform(img).unsqueeze(0)
    
    
    dwt = DWT()
    x_LL, x_HL, x_LH, x_HH = dwt(img)
    
    
    result = (x_HH * 255).squeeze().permute(1, 2, 0).contiguous().numpy().astype(np.int32) # 做高斯处理
    print(result.shape)
    
    plt.imshow(result)
    plt.show()    
    
  