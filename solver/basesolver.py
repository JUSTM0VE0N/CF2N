# -*- encoding: utf-8 -*-
'''
@File    :   basesolver.py
@Time    :   2023/03/28 17:18:50
@Author  :   Liuch(laxio) 
@Version :   1.0
@Source  :   own
@Note    :   for unified train and test
@Contact :   isliuch@yeah.net
'''

# here put the import lib
import os
import torch
import time
from torch.utils.data import DataLoader
from utils.utils import draw_curve_and_save, save_config
from data.data import Load_Trainset

class BaseSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nEpochs = cfg['nEpochs']
        self.checkpoint = cfg['checkpoint']
        self.epoch = 1

        current_time = int(time.time())
        local_time = time.localtime(current_time)
        self.BSTime = time.strftime('%Y-%m-%d-%H_%M_%S', local_time)
        
        if cfg['is_cuda']:
            self.num_worker = cfg['num_worker']
        else:
            self.num_worker = 0
        
        self.training_set = Load_Trainset(cfg['training_set'] + '/' + cfg['satellite4train'])
        self.training_data_loader = DataLoader(dataset=self.training_set, num_workers=self.num_worker, batch_size=cfg['batch_size'], shuffle=True,
                                               pin_memory=True, drop_last=True)
        self.validate_set = Load_Trainset(cfg['training_set'] + '/' + cfg['satellite4valid'])
        self.validate_data_loader = DataLoader(dataset=self.validate_set, num_workers=self.num_worker, batch_size=cfg['batch_size'], shuffle=True,
                                               pin_memory=True, drop_last=True)

        self.records = {'Epoch': [],
                        'SSIM': [],
                        'RMSE': [],
                        'SAM': [],
                        'CC': [],
                        'SCC': [],
                        'PSNR': [],
                        'ERGAS': [],
                        'UIQC': [],
                        'Loss': [],
                        'VLoss': []}
        
        if not os.path.exists(self.checkpoint):
            os.makedirs(self.checkpoint)
    
    def load_checkpoint(self, model_path):
        if os.path.exists(model_path):
            ckpt = torch.load(model_path)
            self.epoch = ckpt['epoch']
            self.records = ckpt['records']
        else:
            raise FileNotFoundError

    def save_checkpoint(self):
        self.ckpt = {
            'epoch': self.epoch,
            'records': self.records
        }

    def train(self):
        raise FileNotFoundError

    def eval(self):
        raise FileNotFoundError

    def run(self):
        while self.epoch <= self.nEpochs:
            self.train()
            self.eval()
            self.save_checkpoint
            self.epoch += 1