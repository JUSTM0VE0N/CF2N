# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 22:26:22 2022
@author: likun
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import model.basic_functions as Funcs
# import torch.fft
        
""" ------------ main fusionnet ------------ """
class fusionnet(nn.Module):
    
    def __init__(
        self, n_iter=6, h_nc=64, in_c=32, out_c=31, m_c=3, nc=[80, 160, 320], nb=1,
        act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose"):
 
        super(fusionnet, self).__init__()
        self.n = n_iter
        self.z = Z_SubNet  (out_c =out_c, m_c=m_c)
        self.h = H_hyperNet(in_c  =3 , out_c=n_iter*3, channel=h_nc)  
        self.x = X_PriorNet(
            in_c=in_c, out_c=out_c, nc=nc, nb=nb,
            act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
      
    def forward(self, Y_h, Y_m, Y_p, lamda_m, lamda_p):
        # Initialization    
        mu     = 0.01
        mu     = torch.tensor(mu).float().view([1, 1, 1, 1]).cuda()
        beta   = torch.sqrt(mu*lamda_p) # ρ = α.mu.lamda_p
        rate_h = Y_p.shape[2]//Y_h.shape[2]
        x      = F.interpolate(Y_h, scale_factor=rate_h, mode ='bilinear')
        # hyper-parameter
        hypers = self.h(torch.cat((mu, lamda_m, beta), dim=1))
        # unfolding    
        for i in range(self.n): 
            z          = self.z(x, Y_h, Y_m, hypers[:, i, ...], hypers[:, i+self.n, ...] )     
            x          = self.x(z, Y_p,      hypers[:, i+self.n*2, ...])
        return x 


""" -------------- -------------- --------------
# (1) Intermediate estimate Z; Gradient-based opt 
# z_k = x_{k-1} - mu{grad(f(.))}
-------------- -------------- -------------- """

class Z_SubNet(nn.Module):
    def __init__(self, out_c =31, m_c=3):
        super(Z_SubNet, self).__init__()

        self.Rm_conv  = Funcs.conv(out_c, m_c, bias=False, mode='1')
        self.RmT_conv = Funcs.conv(m_c, out_c, bias=False, mode='1')

    def forward(self, x, Y_h, Y_m, mu, lamda_m):
        rate_h, rate_m  = x.shape[2]//Y_h.shape[2], x.shape[2]//Y_m.shape[2] 

        XS1         = F.interpolate(x        , scale_factor = 1.0/rate_h , mode ='bilinear')  
        Diff_S1T    = F.interpolate((XS1-Y_h), scale_factor = rate_h     , mode ='bilinear')  
        RXS2        = self.Rm_conv( F.interpolate(x         , scale_factor = 1.0/rate_m , mode ='bilinear'))  
        RTDiff_S2T  = self.RmT_conv(F.interpolate((RXS2-Y_m), scale_factor = rate_m     , mode ='bilinear'))
        Zest = x - mu*Diff_S1T - mu*lamda_m*RTDiff_S2T
        return Zest 


""" -------------- -------------- --------------
# (2) Observation Variable X  && Prior module
#     X  --> Prior pf W   --> obtained X
#     Transfer2 Prior(W)  --> X = W-Rp^Y_p
-------------- -------------- -------------- """ 
class X_PriorNet(nn.Module):
    def __init__(
        self, in_c=32, out_c=31, nc=[80, 160, 320], nb=1, 
        act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        
        super(X_PriorNet, self).__init__() 
        # downsample
        if downsample_mode   == 'avgpool':    downsample_block = Funcs.downsample_avgpool
        elif downsample_mode == 'maxpool':    downsample_block = Funcs.downsample_maxpool
        elif downsample_mode == 'strideconv': downsample_block = Funcs.downsample_strideconv
        else: raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_head  = Funcs.conv(in_c, nc[0], bias=False, mode='C')
        self.m_down1 = Funcs.sequential(
            *[Funcs.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], 
            downsample_block(nc[0], nc[1], bias=False, mode='2')
        )
        self.m_down2 = Funcs.sequential(
            *[Funcs.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], 
            downsample_block(nc[1], nc[2], bias=False, mode='2')
        )
        self.m_body  = Funcs.sequential(
            *[Funcs.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)]
        )
  
        # upsample
        if upsample_mode   == 'upconv':         upsample_block = Funcs.upsample_upconv
        elif upsample_mode == 'pixelshuffle':   upsample_block = Funcs.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':  upsample_block = Funcs.upsample_convtranspose
        else: raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up2  = Funcs.sequential(
            upsample_block(nc[2], nc[1], bias=False, mode='2'), 
            *[Funcs.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)]
        )
        self.m_up1  = Funcs.sequential(
            upsample_block(nc[1], nc[0], bias=False, mode='2'), 
            *[Funcs.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)]
        )
        self.m_tail = Funcs.conv(nc[0], out_c, bias=False, mode='C')


        self.Rp_conv     = nn.Sequential(
                            nn.Conv2d(out_c,  out_c,  3, stride=1, padding=1),
                            nn.ReLU(inplace = True),
                            nn.Conv2d(out_c,  out_c,  3, stride=1, padding=1),)
        self.Rp_hat_conv = nn.Sequential(
                            nn.Conv2d(out_c,  out_c,  3, stride=1, padding=1),
                            nn.ReLU(inplace = True),
                            nn.Conv2d(out_c,  out_c,  3, stride=1, padding=1),)

    def forward(self, Z, Y_p, beta):
        Y_p_copy = Y_p.repeat(1, Z.shape[1], 1, 1) 
        Denoi    = self.Rp_conv(Z) - Y_p_copy  
        Betas    = beta.repeat(Z.size(0), 1, Z.size(2), Z.size(3))         
        W = torch.cat((Denoi ,Betas),  dim = 1)

        ht, wt = W.size()[-2:]
        paddingBottom = int(np.ceil(ht/8)*8-ht)
        paddingRight  = int(np.ceil(wt/8)*8-wt)
        W = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(W)
        W1 = self.m_head(W)
        W2 = self.m_down1(W1)
        W3 = self.m_down2(W2) 
        W  = self.m_body(W3) 
        W  = self.m_up2(W+W3)
        W  = self.m_up1(W+W2)
        W  = self.m_tail(W+W1)
        W = W[..., :ht, :wt]

        Xest = self.Rp_hat_conv(Y_p_copy + W) 
        return Xest
    
        
    
""" -------------- -------------- --------------
# (3) Hyper-parameter module
-------------- -------------- -------------- """ 
class H_hyperNet(nn.Module):
    def __init__(self, in_c=3, out_c=6*3, channel=64):
        super(H_hyperNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_c, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_c, 1, padding=0, bias=True),
                nn.Softplus())
    
    def forward(self, x):
        x = self.mlp(x) + 1e-6
        return x

    