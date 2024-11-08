import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = 'cpu'
else:
    raise 'CUDA is not available'

# NCHW
def Dir(pan):
    shape = pan.shape

    dir_pan1 = np.zeros(shape)
    dir_pan2 = np.zeros(shape)
    dir_pan = np.zeros([shape[0],shape[1]*2,shape[2],shape[3]])

    dir_pan1[:, :, 1:shape[2], :] = pan[:, :, 0:shape[2] - 1, :]    #NCHW
    dir_pan1[:, :, 0, :] = pan[:, :, 0, :]
    dir_pan1 = pan - dir_pan1
    dir_pan2[:, :, :, 1:shape[3]] = pan[:, :, :, 0:shape[3] - 1]
    dir_pan2[:, :, :, 0] = pan[:, :, :, 0]
    dir_pan2 = pan - dir_pan2

    # dir_pan[:,0,:,:]=dir_pan1   #xy
    # dir_pan[:,1,:,:]=dir_pan2
    # dir_pan = dir_pan1+dir_pan2

    dir_pan = np.concatenate((dir_pan1, dir_pan2),axis=1)

    return dir_pan

def sobel_conv(data, channel):
    conv_op_x = nn.Conv2d(channel, channel, 3, stride=1, padding=1, groups=channel, bias=False)
    conv_op_y = nn.Conv2d(channel, channel, 3, stride=1, padding=1, groups=channel, bias=False)
    sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).cuda()

    sobel_kernel_y = torch.tensor([[-1,-2,-1],
                                   [ 0, 0, 0],
                                   [ 1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).cuda()
    conv_op_x.weight.data = sobel_kernel_x
    conv_op_y.weight.data = sobel_kernel_y
    edge_x, edge_y = conv_op_x(data), conv_op_y(data)
    result = 0.5*abs(edge_x) + 0.5*abs(edge_y)
    return result


def prewitt_conv(data, channel):
    conv_op_x = nn.Conv2d(channel, channel, 3, stride=1, padding=1, groups=channel, bias=False)
    conv_op_y = nn.Conv2d(channel, channel, 3, stride=1, padding=1, groups=channel, bias=False)
    sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).cuda()

    sobel_kernel_y = torch.tensor([[-1,-1,-1],
                                   [ 0, 0, 0],
                                   [ 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).cuda()
    conv_op_x.weight.data = sobel_kernel_x
    conv_op_y.weight.data = sobel_kernel_y
    edge_x, edge_y = conv_op_x(data), conv_op_y(data)
    result = 0.5*abs(edge_x) + 0.5*abs(edge_y)
    return result


def laplacian_conv(data, channel):
    conv_op = nn.Conv2d(channel, channel, 3, stride=1, padding=1, groups=channel, bias=False)
    laplacian_kernel = torch.tensor([[ 0,  1, 0],
                                     [ 1, -4, 1],
                                     [ 0,  1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1).cuda()
    conv_op.weight.data = laplacian_kernel
    result = conv_op(data)
    return result

def Edge(pan, isgaussianBlur=1):  #bs 1 size size NCHW
    shape = pan.shape
    batch_size=shape[0]
    size = shape[2]

    robert_pan = np.zeros(shape)
    prewitt_pan = np.zeros(shape)
    sobel_pan = np.zeros(shape)
    laplacian_pan = np.zeros(shape)

    for i in range(0,batch_size):
        pani = np.reshape(pan[i,:,:,:],(size,size))

        # pani = cv.cvtColor(pani, cv.COLOR_BGR2GRAY)
        gaussianBlur = cv.GaussianBlur(pani, (3, 3), 0)  # 高斯滤波
        # ret, binary = cv.threshold(gaussianBlur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) # 阈值处理
        if isgaussianBlur == 0:
            binary = pani
        else:
            binary = gaussianBlur

        normalization = 255.0 / binary.max()
        binary = binary * normalization

        # Sobel算子
        x = cv.Sobel(binary, cv.CV_16S, 1, 0)
        y = cv.Sobel(binary, cv.CV_16S, 0, 1)
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        Sobel = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
        sobel_pan[i,0,:,:] = Sobel/normalization

        binary = binary.astype(np.uint8)

        # 拉普拉斯算法
        dst = cv.Laplacian(binary, cv.CV_16S, ksize=3)
        Laplacian = cv.convertScaleAbs(dst)
        laplacian_pan[i,0,:,:] = (Laplacian)/normalization

        # Roberts算子
        kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely = np.array([[0, -1], [1, 0]], dtype=int)
        x = cv.filter2D(binary, cv.CV_16S, kernelx)
        y = cv.filter2D(binary, cv.CV_16S, kernely)
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        Roberts = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
        robert_pan[i, 0, :, :] = (Roberts)/normalization

        # Prewitt算子
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        x = cv.filter2D(binary, cv.CV_16S, kernelx)
        y = cv.filter2D(binary, cv.CV_16S, kernely)
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        Prewitt = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
        prewitt_pan[i, 0, :, :] = (Prewitt)/normalization

    dir_pan = Dir(pan)

    edge_pan = np.concatenate((dir_pan, robert_pan, prewitt_pan, sobel_pan, laplacian_pan), axis=1)


    return edge_pan

if __name__ == "__main__":
    img = cv.imread("../lakers.jpg", cv.IMREAD_COLOR)
    img2 = torch.from_numpy(img).permute([2, 0, 1]).unsqueeze(0)

    edge = laplacian_conv(img2.to(torch.float32), 3).squeeze(0).detach().numpy().transpose([1, 2, 0])
    cv.imshow('d', edge)
    cv.waitKey(0)