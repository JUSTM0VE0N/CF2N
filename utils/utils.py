# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2023/07/03 02:16:18
@Author  :   Liuch(laxio) 
@Version :   1.0
@Source  :   own
@Note    :   some utils for storage
@Contact :   isliuch@yeah.net
'''

# here put the import lib
import os
import math
import torch
import cv2
import shutil
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
# from utils.vgg import VGG
import torch.nn.functional as F

def make_optimizer(opt_type, cfg, params):
    if opt_type == "ADAM":
        optimizer = torch.optim.Adam(params, lr=cfg['schedule']['lr'], betas=(cfg['schedule']['beta1'], cfg['schedule']['beta2']), eps=cfg['schedule']['epsilon'])
    elif opt_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=cfg['schedule']['lr'], momentum=cfg['schedule']['momentum'])
    elif opt_type == "RMSprop":
        optimizer = torch.optim.RMSprop(params, lr=cfg['schedule']['lr'], alpha=cfg['schedule']['alpha'])
    else:
        raise ValueError
    return optimizer

def make_loss(loss_type):
    # loss = {}
    if loss_type == "MSE":
        loss = nn.MSELoss()
    elif loss_type == "L1":
        loss = nn.L1Loss()
    else:
        raise ValueError
    return loss

def get_path(subdir):
    return os.path.join(subdir)

def save_config(cfg, time, log):
    open_type = 'a' if os.path.exists(get_path('./logs/' + cfg['model'] + '/' + str(time) + '/records.txt'))else 'w'
    log_file = open(get_path('./logs/' + cfg['model'] + '/' + str(time) + '/records.txt'), open_type)
    log_file.write(str(log) + '\n')

def save_net_config(cfg, time, log):
    Folder=get_path('./logs/' + cfg['model'] + '/' + str(time) + '/')
    if not os.path.exists(Folder):
        os.makedirs(Folder)
    log_file = open(get_path(Folder+'/net.txt'), 'w')
    log_file.write(str(log) + '\n')

def save_net_py(cfg, time, py):
    shutil.copyfile(os.path.join('./models', py+'.py'), os.path.join('./logs/' + cfg['model'] + '/' + str(time), py+'.py'))
    
def draw_curve_and_save(x, y, title, filename, precision):
    if not isinstance(x, np.ndarray):
        x = np.array(x).astype(np.int32)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.set_title(title)

    max_y = np.ceil(y.max() / precision) * precision
    min_y = np.floor(y.min() / precision) * precision
    major_y_step = (max_y - min_y) / 10
    if major_y_step < 0.1:
        major_y_step = 0.1
    ax.yaxis.set_major_locator(MultipleLocator(major_y_step))
    ax.yaxis.set_minor_locator(MultipleLocator(major_y_step))
    ax.yaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='both')
    if (x.shape[0] >= 2):
        axis_range = [x.min(), x.man(), min_y, max_y]
        ax.axis(axis_range)
    ax.plot(x, y)
    plt.savefig(filename)