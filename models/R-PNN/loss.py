import torch.nn as nn
import torch
from math import ceil, floor
import numpy as np
from tools.cross_correlation import xcorr_torch as ccorr


class SpectralLoss(nn.Module):
    def __init__(self, mtf, ratio, device):

        # Class initialization
        super(SpectralLoss, self).__init__()
        kernel = mtf
        # Parameters definition
        self.nbands = kernel.shape[-1]
        self.device = device
        self.ratio = ratio

        # Conversion of filters in Tensor
        self.pad = floor((kernel.shape[0] - 1) / 2)

        self.cut_border = kernel.shape[0] // 2 // ratio

        kernel = np.moveaxis(kernel, -1, 0)
        kernel = np.expand_dims(kernel, axis=1)

        kernel = torch.from_numpy(kernel).type(torch.float32)

        # DepthWise-Conv2d definition
        self.depthconv = nn.Conv2d(in_channels=self.nbands,
                                   out_channels=self.nbands,
                                   groups=self.nbands,
                                   kernel_size=kernel.shape,
                                   bias=False)

        self.depthconv.weight.data = kernel
        self.depthconv.weight.requires_grad = False

        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, outputs, labels):

        outputs = self.depthconv(outputs)
        outputs = outputs[:, :, 3::self.ratio, 3::self.ratio]

        loss_value = self.loss(outputs, labels[:, :, self.cut_border:-self.cut_border, self.cut_border:-self.cut_border])

        return loss_value

class StructuralLoss(nn.Module):

    def __init__(self, sigma):
        # Class initialization
        super(StructuralLoss, self).__init__()

        # Parameters definition:
        self.scale = ceil(sigma / 2)

    def forward(self, outputs, labels, xcorr_thr):
        x_corr = torch.clamp(ccorr(outputs, labels, self.scale), min=-1)
        x = 1.0 - x_corr

        with torch.no_grad():
            loss_cross_corr_wo_thr = torch.mean(x)

        worst = x.gt(xcorr_thr)
        y = x * worst
        loss_cross_corr = torch.mean(y)

        return loss_cross_corr, loss_cross_corr_wo_thr.item()
