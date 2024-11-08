# -*- encoding: utf-8 -*-
'''
@File    :   solver.py
@Time    :   2023/03/29 12:09:21
@Author  :   Liuch(laxio) 
@Version :   1.0
@Source  :   own
@Note    :   for unified train
@Contact :   isliuch@yeah.net
'''

# here put the import lib
from configparser import Interpolation
import torch.nn.functional as F
import os
import importlib
from importlib import import_module
import shutil
import torch
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from solver.basesolver import BaseSolver
from utils.utils import make_optimizer, make_loss, save_config, save_net_config, save_net_py
from utils.metrics import *
from utils.config import save_yml
# from collections import Iterable
# from PIL import Image, IamgeOps
# import torchvision.transforms.functional as fn
 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Solver(BaseSolver):
    def __init__(self, cfg):
        super(Solver, self).__init__(cfg)
        self.init_epoch = self.cfg['schedule']
        net_name = self.cfg['model']
        lib = importlib.import_module('models.' + net_name)
        net = lib.Network

        # self.model = net(1, 8, args = self.cfg)  #mucnn
        self.model = net(args=self.cfg)  #MMALPNet
        # self.model = net(args=self.cfg)  # other models
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        device_ids = [0, 1]
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids).cuda()

        self.optimizer = make_optimizer(self.cfg['schedule']['optimizer'], cfg, self.model.parameters())
        self.milestones = list(map(lambda x: int(x), self.cfg['schedule']['decay'].split('-')))
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, self.milestones, gamma=self.cfg['schedule']['gamma'], last_epoch=-1)
        self.loss = make_loss(self.cfg['schedule']['loss'])
        self.down = ()
        self.log_name = self.cfg['model'] + '_' + str(self.BSTime)
        # save log
        self.writer = SummaryWriter('logs/' + self.cfg['model'] + '/' + str(self.log_name))
        save_net_config(cfg, self.log_name, self.model)
        save_net_py(cfg, self.log_name, net_name)
        save_yml(cfg, os.path.join('logs/' + cfg['model'] + '/' + str(self.log_name), 'config.yml'))
        save_config(cfg, self.log_name, 'Train dataset has {} images and {} batches.'.format(len(self.training_set), len(self.training_data_loader)))
        save_config(cfg, self.log_name, 'Val dataset has {} images and {} batches.'.format(len(self.validate_set), len(self.validate_data_loader)))
        save_config(cfg, self.log_name, 'Model parameters: '+ str(sum(param.numel() for param in self.model.parameters())))

    def train(self):
        with tqdm(total=len(self.training_data_loader), miniters=1,
                  desc='\033[32mTraining Epoch: [{}/{}]\033[0m'.format(self.epoch, self.nEpochs), colour='green') as t:
            epoch_loss = []
            for iteration, batch in enumerate(self.training_data_loader,1):
                gt, lms, ms, pan = Variable(batch[0], requires_grad=False), \
                    Variable(batch[1]), \
                    Variable(batch[2]), \
                    Variable(batch[3])

                if self.cuda:
                    gt, lms, ms, pan = Variable(batch[0], requires_grad=False).cuda(), \
                        Variable(batch[1]).cuda(), \
                        Variable(batch[2]).cuda(), \
                        Variable(batch[3]).cuda()

                gt_d2 = F.interpolate(gt, size=[gt.size(2) // 2, gt.size(2) // 2], mode='bicubic', align_corners=True).cuda()
                gt_d4 = F.interpolate(gt, size=[gt.size(2) // 4, gt.size(2) // 4], mode='bicubic', align_corners=True).cuda()

                # ms1 = F.interpolate(ms_image, size=[ms_image.size(2) // 2, ms_image.size(2) // 2], mode='bicubic', align_corners=True)
                # ms2 = F.interpolate(ms_image, size=[ms_image.size(2) // 4, ms_image.size(2) // 4], mode='bicubic', align_corners=True)
                self.optimizer.zero_grad()               
                self.model.train()
                # MUCNN
                out1, out2, out3, ms_u4 = self.model(ms.float(), pan.float())
                loss_d4 = self.loss(out1.float(), gt_d4.float())
                loss_d2 = self.loss(out2.float(), gt_d2.float())
                loss_o = self.loss(out3.float(), gt.float())
                loss_ms = self.loss(ms_u4.float(), lms.float())
                loss = self.cfg['hyperparameter']['lambda1'] * loss_d4 + self.cfg['hyperparameter']['lambda2'] * loss_d2 + self.cfg['hyperparameter']['lambda3'] * loss_o + self.cfg['hyperparameter']['lambda4'] * loss_ms

                epoch_loss.append(loss.item())

                t.set_postfix_str("\033[32mPer batch training loss {:.7f}\033[0m".format(loss.item()))
                t.update()
                # loss.requires_grad_(True)
                loss.backward()
                self.optimizer.step()
            # re_loss = epoch_loss / len(self.train_loader)
            self.records['Loss'].append(np.nanmean(np.array(epoch_loss)))
            # self.writer.add_image('image1', gt[0], self.epoch)
            # self.writer.add_image('image2', y[0], self.epoch)
            # self.writer.add_image('image2', out3[0], self.epoch)
            # self.writer.add_image('image3', pan[0], self.epoch)
            save_config(self.cfg, self.log_name, 'Training Epoch {}: Loss={:.7f}'.format(self.epoch, self.records['Loss'][-1]))
            self.writer.add_scalar('Loss_epoch', self.records['Loss'][-1], self.epoch)

    def eval(self):
        with tqdm(total=len(self.validate_data_loader), miniters=1,
                desc='\033[32mVal Epoch: [{}/{}]\033[0m'.format(self.epoch, self.nEpochs), colour='green') as t1:
            epoch_val_loss = []
            # ssim_list = []
            # rmse_list = []
            # sam_list = []
            # cc_list = []
            # scc_list = []
            # psnr_list = []
            ergas_list = []
            uiqc_list = []
            for iteration, batch in enumerate(self.validate_data_loader, 1):

                gt, lms, ms, pan = Variable(batch[0], requires_grad=False), \
                    Variable(batch[1]), \
                    Variable(batch[2]), \
                    Variable(batch[3])

                if self.cuda:
                    gt, lms, ms, pan = Variable(batch[0], requires_grad=False).cuda(), \
                        Variable(batch[1]).cuda(), \
                        Variable(batch[2]).cuda(), \
                        Variable(batch[3]).cuda()

                    gt_d2 = F.interpolate(gt, size=[gt.size(2) // 2, gt.size(2) // 2], mode='bicubic', align_corners=True).cuda()
                    gt_d4 = F.interpolate(gt, size=[gt.size(2) // 4, gt.size(2) // 4], mode='bicubic', align_corners=True).cuda()

                self.model.eval()
                with torch.no_grad():
                    out1, out2, out3, ms_u4 = self.model(ms.float(), pan.float())
                    loss_d4 = self.loss(out1.float(), gt_d4.float())
                    loss_d2 = self.loss(out2.float(), gt_d2.float())
                    loss_o = self.loss(out3.float(), gt.float())
                    loss_ms = self.loss(ms_u4.float(), lms.float())
                    loss = self.cfg['hyperparameter']['lambda1'] * loss_d4 + self.cfg['hyperparameter']['lambda2'] * loss_d2 + self.cfg['hyperparameter']['lambda3'] * loss_o + self.cfg['hyperparameter']['lambda4'] * loss_ms
                    epoch_val_loss.append(loss.item())

                # batch_ssim = []
                # batch_rmse = []
                # batch_sam = []
                # batch_cc = []
                # batch_scc = []
                # batch_psnr = []
                batch_ergas = []
                batch_uiqc = []
                
                # y = y[:,0:3,:,:]
                # ms_image=ms_image[:,0:3,:,:]
                for i in range(out3.shape[0]):
                    predict_out = (out3[i, :, :, :].cpu().detach().numpy().transpose((1, 2, 0)))
                    ground_truth = (gt[i, :, :, :].cpu().detach().numpy().transpose((1, 2, 0)))

                    # ssim = SSIM_numpy(ground_truth, predict_out, data_range=1)
                    # rmse = RMSE_numpy(ground_truth, predict_out)
                    # sam = SAM_numpy(ground_truth, predict_out)
                    # cc = CC_numpy(ground_truth, predict_out)
                    # scc = SCC_numpy(ground_truth, predict_out)
                    # psnr = MPSNR_numpy(ground_truth, predict_out, data_range=1)
                    ergas = ERGAS_numpy(ground_truth, predict_out)
                    uiqc = UIQC_numpy(ground_truth, predict_out)
                    
                    # batch_ssim.append(ssim)
                    # batch_rmse.append(rmse)
                    # batch_sam.append(sam)
                    # batch_cc.append(cc)
                    # batch_scc.append(scc)
                    # batch_psnr.append(psnr)
                    batch_ergas.append(ergas)
                    batch_uiqc.append(uiqc)

                # avg_ssim = np.array(batch_ssim).mean()
                # avg_rmse = np.array(batch_rmse).mean()
                # avg_sam = np.array(batch_sam).mean()
                # avg_cc = np.array(batch_cc).mean()
                # avg_scc = np.array(batch_scc).mean()
                # avg_psnr = np.array(batch_psnr).mean()
                avg_ergas = np.array(batch_ergas).mean()
                avg_uiqc = np.array(batch_uiqc).mean()  
                
                # ssim_list.extend(batch_ssim)
                # rmse_list.extend(batch_rmse)
                # sam_list.extend(batch_sam)
                # cc_list.extend(batch_cc)
                # scc_list.extend(batch_scc)
                # psnr_list.extend(batch_psnr)
                ergas_list.extend(batch_ergas)
                uiqc_list.extend(batch_uiqc)

                # t1.set_postfix_str('\033[32mPer batch validate loss: {:.7f}, SSIM: {:.4f}, RMSE: {:.4f}, SAM: {:.4f}, CC: {:.4f}, SCC: {:.4f}, PSNR: {:.4f}, ERGAS: {:.4f}, UIQC: {:.4f}\033[0m'.format(loss.item(), avg_ssim, avg_rmse, avg_sam, avg_cc, avg_scc, avg_psnr, avg_ergas, avg_uiqc))
                t1.set_postfix_str(
                    '\033[32mPer batch validate loss: {:.7f}, ERGAS: {:.4f}, UIQC: {:.4f}\033[0m'.format(
                        loss.item(), avg_ergas, avg_uiqc))
                t1.update()
            self.records['Epoch'].append(self.epoch)
            self.records['VLoss'].append(np.nanmean(np.array(epoch_val_loss)))
            # self.records['SSIM'].append(np.array(ssim_list).mean())
            # self.records['RMSE'].append(np.array(rmse_list).mean())
            # self.records['SAM'].append(np.array(sam_list).mean())
            # self.records['CC'].append(np.array(cc_list).mean())
            # self.records['SCC'].append(np.array(scc_list).mean())
            # self.records['PSNR'].append(np.array(psnr_list).mean())
            self.records['ERGAS'].append(np.array(ergas_list).mean())
            self.records['UIQC'].append(np.array(uiqc_list).mean())

            # save_config(self.cfg, self.log_name, 'Validate Epoch {}: TLoss: {:.7f}, SSIM: {:.4f}, RMSE: {:.4f}, SAM: {:.4f}, CC: {:.4f}, SCC: {:.4f}, PSNR: {:.4f}, ERGAS: {:.4f}, UIQC: {:.4f}'.format\
            #     (self.epoch, self.records['VLoss'][-1], self.records['SSIM'][-1], self.records['RMSE'][-1], self.records['SAM'][-1],self.records['CC'][-1], self.records['SCC'][-1], self.records['PSNR'][-1], self.records['ERGAS'][-1],self.records['UIQC'][-1]))
            save_config(self.cfg, self.log_name,
                        'Validate Epoch {}: TLoss: {:.7f}, ERGAS: {:.4f}, UIQC: {:.4f}'.format \
                            (self.epoch, self.records['VLoss'][-1], self.records['ERGAS'][-1], self.records['UIQC'][-1]))
            self.writer.add_scalar('Val_loss_epoch', self.records['VLoss'][-1], self.epoch)
            # self.writer.add_scalar('SSIM_epoch', self.records['SSIM'][-1], self.epoch)
            # self.writer.add_scalar('RMSE_epoch', self.records['RMSE'][-1], self.epoch)
            # self.writer.add_scalar('SAM_epoch', self.records['SAM'][-1], self.epoch)
            # self.writer.add_scalar('CC_epoch', self.records['CC'][-1], self.epoch)
            # self.writer.add_scalar('SCC_epoch', self.records['SCC'][-1], self.epoch)
            # self.writer.add_scalar('PSNR_epoch', self.records['PSNR'][-1], self.epoch)
            self.writer.add_scalar('ERGAS_epoch', self.records['ERGAS'][-1], self.epoch)
            self.writer.add_scalar('UIQC_epoch', self.records['UIQC'][-1], self.epoch)

    def check_gpu(self):
        self.cuda = self.cfg['is_cuda']
        torch.manual_seed(self.cfg['seed'])
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run with CPU")
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed'])
            torch.manual_seed(self.cfg['seed'])
            torch.cuda.manual_seed(self.cfg['seed'])
            torch.cuda.manual_seed_all(self.cfg['seed'])
            cudnn.benchmark = True   
            cudnn.deterministic = True
              
            gups_list = self.cfg['gpus']
            self.gpu_ids = []
            for str_id in gups_list:
                gid = int(str_id)
                if gid >= 0:
                    self.gpu_ids.append(gid)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_ids)
            if len(self.gpu_ids) > 2:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids) 
            else:
                torch.cuda.set_device(self.gpu_ids[0]) 
                self.loss = self.loss.cuda(self.gpu_ids[0])
                self.model = self.model.cuda(self.gpu_ids[0])
            

    def check_pretrained(self):
        checkpoint = os.path.join(self.cfg['pretrain']['pre_folder'], self.cfg['satellite'], self.log_name, self.cfg['pretrain']['pre_source'])
        if os.path.exists(checkpoint):
            self.model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['net'])
            self.epoch = torch.load(checkpoint, map_location=lambda storage, loc: storage)['epoch']
            for _ in range(self.epoch-1): self.scheduler.step()
            self.optimizer.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['optimizer'])
            if self.epoch > self.nEpochs:
                raise Exception("Pretrain epoch must less than the max epoch!")
        else:
            raise Exception("Pretrain path error!")

    def save_checkpoint(self):
        super(Solver, self).save_checkpoint()
        self.ckpt['net'] = self.model.state_dict()
        self.ckpt['optimizer'] = self.optimizer.state_dict()
        self.ckpt_dir = self.cfg['checkpoint'] + '/' + self.cfg['model'] + '/' + self.cfg['satellite'] + '/' + self.log_name
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        torch.save(self.ckpt, self.ckpt_dir + '/' + '{}.pth'.format(self.epoch))

        # if self.cfg['save_best']:
        #     if self.records['UIQC'] != [] and self.records['UIQC'][-1] == np.array(self.records['UIQC']).max():
        #         shutil.copy(os.path.join(self.ckpt_dir, 'latest.pth'),
        #                     os.path.join(self.ckpt_dir, 'best.pth'))

    def run(self):
        self.check_gpu()
        if self.cfg['pretrain']['pretrained']:
            self.check_pretrained()
        try:
            while self.epoch <= self.nEpochs:
                self.train()
                self.eval()
                self.save_checkpoint()
                self.scheduler.step()
                self.epoch += 1
        except KeyboardInterrupt:
            self.save_checkpoint()
        save_config(self.cfg, self.log_name, 'Training done.')



    def down(self,x):
        self.down()
        self.max_pool_conv = nn.MaxPool2d(2)
        return self.max_pool_conv(x)