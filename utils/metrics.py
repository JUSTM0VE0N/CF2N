# -*- encoding: utf-8 -*-
'''
@File    :   metrics.py
@Time    :   2023/03/29 12:14:42
@Author  :   Liuch(laxio) 
@Version :   1.0
@Source  :   own
@Note    :   metrics for evalution
@Contact :   isliuch@yeah.net
'''

# here put the import lib
import numpy as np
from numpy.linalg import norm
import cv2
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.ndimage.filters import sobel, convolve
from scipy.stats import pearsonr
import sewar as sewar_api


# numpy version
def SSIM_numpy(x_true, x_pred, data_range, sewar=False):
    r"""
    SSIM(Structural Similarity)，结构相似性，是一种衡量两幅图像相似度的指标。
    结构相似性的范围为-1到1。当两张图像一模一样时，SSIM的值等于1。
    结构相似度指数从图像组成的角度将结构信息定义为独立于亮度、对比度的，反映场景中物体结构的属性，
    并将失真建模为亮度、对比度和结构三个不同因素的组合。
    用均值作为亮度的估计，标准差作为对比度的估计，协方差作为结构相似程度的度量。
    Args:
        x_true (np.ndarray): target image, shape like [H, W, C]
        x_pred (np.ndarray): predict image, shape like [H, W, C]
        data_range (int): max_value of the image
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: SSIM value
    """
    if sewar:
        return sewar_api.ssim(x_true, x_pred, MAX=data_range)[0]

    return structural_similarity(x_true, x_pred, data_range=data_range, channel_axis=2)


def MPSNR_numpy(x_true, x_pred, data_range):
    r"""
    Args:
        x_true (np.ndarray): target image, shape like [H, W, C]
        x_pred (np.ndarray): predict image, shape like [H, W, C]
        data_range (int): max_value of the image
    Returns:
        float: Mean PSNR value
    """

    tmp = []
    for c in range(x_true.shape[-1]):
        tmp.append(peak_signal_noise_ratio(x_true[:, :, c], x_pred[:, :, c], data_range=data_range))
    return np.mean(tmp)


def SAM_numpy(x_true, x_pred, sewar=False):
    r"""
    Look at paper:
    `Discrimination among semiarid landscape endmembers using the spectral angle mapper (sam) algorithm` for details
    SAM用来计算两个数组之间的相似性，其计算结果可看作两数组之间余弦角
    输出结果值越小表示两个数组越匹配，相似度越高。反之，表示两数组距离越大，相似度越小。
    Args:
        x_true (np.ndarray): target image, shape like [H, W, C]
        x_pred (np.ndarray): predict image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: SAM value
    """
    if sewar:
        return sewar_api.sam(x_true, x_pred)

    assert x_true.ndim == 3 and x_true.shape == x_pred.shape
    dot_sum = np.sum(x_true * x_pred, axis=2)
    norm_true = norm(x_true, axis=2)
    norm_pred = norm(x_pred, axis=2)
    np.seterr(divide='ignore', invalid='ignore')
    cos_value = dot_sum / norm_pred / norm_true
    eps = 1e-6
    if 1.0 < cos_value.any() < 1.0 + eps:
        cos_value = 1.0
    elif -1.0 - eps < cos_value.any() < -1.0:
        cos_value = -1.0
    res = np.arccos(cos_value)
    is_nan = np.nonzero(np.isnan(res))
    # 返回的是x中的不为0的元素坐标
    # isnan返回的是数组对应的相同大小的布尔型数组
    for (x, y) in zip(is_nan[0], is_nan[1]):
        res[x, y] = 0
    sam = np.mean(res)
    return sam * 180 / np.pi


def SCC_numpy(ms, ps, sewar=False):
    r"""
    Look at paper:
    `A wavelet transform method to merge Landsat TM and SPOT panchromatic data` for details

    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: SCC value
    """
    if sewar:
        return sewar_api.scc(ms, ps)

    ps_sobel = sobel(ps, mode='constant')
    ms_sobel = sobel(ms, mode='constant')
    scc = 0.0
    for i in range(ms.shape[2]):
        a = (ps_sobel[:, :, i]).reshape(ms.shape[0] * ms.shape[1])
        b = (ms_sobel[:, :, i]).reshape(ms.shape[0] * ms.shape[1])
        scc += pearsonr(a, b)[0]
    return scc / ms.shape[2]


def CC_numpy(ms, ps):
    r"""
    相关系数CC
    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
    Returns:
        float: CC value
    """

    cc = 0.0
    for i in range(ms.shape[2]):
        a = (ps[:, :, i]).reshape(ms.shape[0] * ms.shape[1])
        b = (ms[:, :, i]).reshape(ms.shape[0] * ms.shape[1])
        cc += pearsonr(a, b)[0]
    return cc / ms.shape[2]


def Q4_numpy(ms, ps):
    r"""
    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
    Returns:
        float: Q4 value
    """

    def conjugate(a):
        sign = -1 * np.ones(a.shape)
        sign[0, :] = 1
        return a * sign

    def product(a, b):
        a = a.reshape(a.shape[0], 1)
        b = b.reshape(b.shape[0], 1)
        R = np.dot(a, b.transpose())
        r = np.zeros(4)
        r[0] = R[0, 0] - R[1, 1] - R[2, 2] - R[3, 3]
        r[1] = R[0, 1] + R[1, 0] + R[2, 3] - R[3, 2]
        r[2] = R[0, 2] - R[1, 3] + R[2, 0] + R[3, 1]
        r[3] = R[0, 3] + R[1, 2] - R[2, 1] + R[3, 0]
        return r

    imps = np.copy(ps)
    imms = np.copy(ms)
    vec_ps = imps.reshape(imps.shape[1] * imps.shape[0], imps.shape[2])  # (W*H, C)
    vec_ps = vec_ps.transpose(1, 0)  # (C, W*H)
    vec_ms = imms.reshape(imms.shape[1] * imms.shape[0], imms.shape[2])
    vec_ms = vec_ms.transpose(1, 0)  # (C, W*H)
    m1 = np.mean(vec_ps, axis=1)
    d1 = (vec_ps.transpose(1, 0) - m1).transpose(1, 0)  # (C, W*H)
    s1 = np.mean(np.sum(d1 * d1, axis=0))
    m2 = np.mean(vec_ms, axis=1)
    d2 = (vec_ms.transpose(1, 0) - m2).transpose(1, 0)  # (C, W*H)
    s2 = np.mean(np.sum(d2 * d2, axis=0))
    Sc = np.zeros(vec_ms.shape)  # (C, W*H)
    d2 = conjugate(d2)
    for i in range(vec_ms.shape[1]):
        Sc[:, i] = product(d1[:, i], d2[:, i])
    C = np.mean(Sc, axis=1)
    Q4 = 4 * np.sqrt(np.sum(m1 * m1) * np.sum(m2 * m2) * np.sum(C * C)) / (s1 + s2) / (
                np.sum(m1 * m1) + np.sum(m2 * m2))
    return Q4


def RMSE_numpy(ms, ps, sewar=False):
    r"""
    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: RMSE value
    """
    if sewar:
        return sewar_api.rmse(ms, ps)

    d = (ms - ps) ** 2
    rmse = np.sqrt(np.sum(d) / (d.shape[0] * d.shape[1]))
    return rmse


def ERGAS_numpy(ms, ps, ratio=0.25, sewar=False):
    r"""
    Look at paper:
    `Quality of high resolution synthesised images: Is there a simple criterion?` for details
    相对全局无纲量误差
    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: ERGAS value
    """
    if sewar:
        return sewar_api.ergas(ms, ps)

    m, n, d = ms.shape
    summed = 0.0
    for i in range(d):
        summed += (RMSE_numpy(ms[:, :, i], ps[:, :, i])) ** 2 / np.mean(ps[:, :, i]) ** 2
    ergas = 100 * ratio * np.sqrt(summed / d)
    return ergas


def UIQC_numpy(ms, ps, sewar=False):
    r"""
    Args:
        ms (np.ndarray): target image, shape like [H, W, C]
        ps (np.ndarray): predict image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: UIQC value
    """
    if sewar:
        return sewar_api.uqi(ms, ps)

    l = ms.shape[2]
    uiqc = 0.0
    for i in range(l):
        uiqc += QIndex_numpy(ms[:, :, i], ps[:, :, i])
    return uiqc / l


def QIndex_numpy(a, b):
    r"""
    Look at paper:
    `A universal image quality index` for details

    Args:
        a (np.ndarray): one-channel image, shape like [H, W]
        b (np.ndarray): one-channel image, shape like [H, W]
    Returns:
        float: Q index value
    """
    a = a.reshape(a.shape[0] * a.shape[1])
    b = b.reshape(b.shape[0] * b.shape[1])
    temp = np.cov(a, b)
    d1 = temp[0, 0]
    cov = temp[0, 1]
    d2 = temp[1, 1]
    m1 = np.mean(a)
    m2 = np.mean(b)
    Q = 4 * cov * m1 * m2 / (d1 + d2 + 1e-21) / (m1 ** 2 + m2 ** 2 + 1e-21)

    return Q


def D_lambda_numpy(l_ms, ps, sewar=False):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (np.ndarray): LR MS image, shape like [H, W, C]
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: D_lambda value
    """
    if sewar:
        return sewar_api.d_lambda(l_ms, ps)

    L = ps.shape[2]
    sum = 0.0
    for i in range(L):
        for j in range(L):
            if j != i:
                sum += np.abs(QIndex_numpy(ps[:, :, i], ps[:, :, j]) - QIndex_numpy(l_ms[:, :, i], l_ms[:, :, j]))
    return sum / L / (L - 1)


def D_s_numpy(l_ms, pan, ps, sewar=False):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (np.ndarray): LR MS image, shape like [H, W, C]
        pan (np.ndarray): pan image, shape like [H, W]
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
        sewar (bool): use the api from sewar, Default: False
    Returns:
        float: D_s value
    """
    if sewar:
        return sewar_api.d_s(pan, l_ms, ps)

    L = ps.shape[2]
    # cv2.pyrDown() 从一个高分辨率大尺寸的图像向上构建一个金字塔（尺寸变小，分辨率降低）
    l_pan = cv2.pyrDown(pan)
    l_pan = cv2.pyrDown(l_pan)
    sum = 0.0
    for i in range(L):
        sum += np.abs(QIndex_numpy(ps[:, :, i], pan) - QIndex_numpy(l_ms[:, :, i], l_pan))
    return sum / L


def FCC_numpy(pan, ps):
    r"""
    Look at paper:
    `A wavelet transform method to merge landsat TM and SPOT panchromatic data` for details

    Args:
        pan (np.ndarray): pan image, shape like [H, W]
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
    Returns:
        float: FCC value
    """
    k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    fcc = []
    for i in range(ps.shape[2]):
        a = convolve(ps[:, :, i], k, mode='constant').reshape(-1)
        b = convolve(pan, k, mode='constant').reshape(-1)
        fcc.append(pearsonr(b, a)[0])  # 计算两个数组的相关系数 输出的第一个值为相关系数；第二个值为p值，该值越小表明相关系数越显著
    return np.max(fcc)


def SF_numpy(ps):
    r"""
    Look at paper:
    `Review of pixel-level image fusion` for details

    Args:
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
    Returns:
        float: SF value
    """
    f_row = np.mean((ps[:, 1:] - ps[:, :-1]) * (ps[:, 1:] - ps[:, :-1]))
    f_col = np.mean((ps[1:, :] - ps[:-1, :]) * (ps[1:, :] - ps[:-1, :]))
    return np.sqrt(f_row + f_col)


def SD_numpy(ps):
    r"""
    Look at paper:
    `A novel metric approach evaluation for the spatial enhancement of pansharpened images` for details

    Args:
        ps (np.ndarray): pan-sharpened image, shape like [H, W, C]
    Returns:
        float: SD value
    """
    SD = 0.0
    for i in range(ps.shape[2]):
        SD += np.std(ps[:, :, i].reshape(-1))
    return SD / ps.shape[2]


# torch version
def SAM_torch(x_true, x_pred):
    r"""
    Look at paper:
    `Discrimination among semiarid landscape endmembers using the spectral angle mapper (sam) algorithm` for details

    Args:
        x_true (torch.Tensor): target images, shape like [N, C, H, W]
        x_pred (torch.Tensor): predict images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean SAM value of n images
    """
    dot_sum = torch.sum(x_true * x_pred, dim=1)
    norm_true = torch.norm(x_true, dim=1)
    norm_pred = torch.norm(x_pred, dim=1)
    a = torch.Tensor([1]).to(x_true.device, dtype=x_true.dtype)
    b = torch.Tensor([-1]).to(x_true.device, dtype=x_true.dtype)
    res = dot_sum / norm_pred / norm_true
    res = torch.max(torch.min(res, a), b)
    res = torch.acos(res) * 180 / 3.1415926
    sam = torch.mean(res)
    return sam


def sobel_torch(im):
    r"""
    Args:
        im (torch.Tensor): images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: images after sobel filter
    """
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = torch.Tensor(sobel_kernel).to(im.device, dtype=im.dtype)
    return F.conv2d(im, weight)


def SCC_torch(x, y):
    r"""
    Args:
        x (torch.Tensor): target images, shape like [N, C, H, W]
        y (torch.Tensor): predict images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean SCC value of n images
    """
    x = sobel_torch(x)
    y = sobel_torch(y)

    vx = x - torch.mean(x, dim=(2, 3), keepdim=True)
    vy = y - torch.mean(y, dim=(2, 3), keepdim=True)
    scc = torch.sum(vx * vy, dim=(2, 3)) / torch.sqrt(torch.sum(vx * vx, dim=(2, 3))) / torch.sqrt(
        torch.sum(vy * vy, dim=(2, 3)))
    return torch.mean(scc)


def QIndex_torch(a, b, eps=1e-8):
    r"""
    Look at paper:
    `A universal image quality index` for details

    Args:
        a (torch.Tensor): one-channel images, shape like [N, H, W]
        b (torch.Tensor): one-channel images, shape like [N, H, W]
    Returns:
        torch.Tensor: Q index value of all images
    """
    E_a = torch.mean(a, dim=(1, 2))
    E_a2 = torch.mean(a * a, dim=(1, 2))
    E_b = torch.mean(b, dim=(1, 2))
    E_b2 = torch.mean(b * b, dim=(1, 2))
    E_ab = torch.mean(a * b, dim=(1, 2))
    var_a = E_a2 - E_a * E_a
    var_b = E_b2 - E_b * E_b
    cov_ab = E_ab - E_a * E_b
    return torch.mean(4 * cov_ab * E_a * E_b / ( (var_a + var_b) * (E_a ** 2 + E_b ** 2) + eps) )


def D_lambda_torch(l_ms, ps):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (torch.Tensor): LR MS images, shape like [N, C, H, W]
        ps (torch.Tensor): pan-sharpened images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean D_lambda value of n images
    """
    L = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)
    for i in range(L):
        for j in range(L):
            if j != i:
                sum += torch.abs(QIndex_torch(ps[:, i, :, :], ps[:, j, :, :]) - QIndex_torch(l_ms[:, i, :, :], l_ms[:, j, :, :]))
    return sum / L / (L - 1)


def D_s_torch(l_ms, pan, l_pan, ps):
    r"""
    Look at paper:
    `Multispectral and panchromatic data fusion assessment without reference` for details

    Args:
        l_ms (torch.Tensor): LR MS images, shape like [N, C, H, W]
        pan (torch.Tensor): PAN images, shape like [N, C, H, W]
        l_pan (torch.Tensor): LR PAN images, shape like [N, C, H, W]
        ps (torch.Tensor): pan-sharpened images, shape like [N, C, H, W]
    Returns:
        torch.Tensor: mean D_s value of n images
    """
    L = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)
    for i in range(L):
        sum += torch.abs(QIndex_torch(ps[:, i, :, :], pan[:, 0, :, :]) - QIndex_torch(l_ms[:, i, :, :], l_pan[:, 0, :, :]))
    return sum / L
