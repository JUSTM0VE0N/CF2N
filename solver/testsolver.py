# -*- encoding: utf-8 -*-
'''
@File    :   testsolver.py
@Time    :   2023/03/30 12:10:13
@Author  :   Liuch(laxio) 
@Version :   1.0
@Source  :   own
@Note    :   for unified test
@Contact :   isliuch@yeah.net
'''

# here put the import lib
import os, torch, time, cv2, importlib
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable 
import numpy as np
from PIL import Image
import scipy.io as sio

from solver.basesolver import BaseSolver
from solver.solver import Solver
from data.data import Load_RRTset, Load_FRTset
from utils.metrics import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Testsolver(BaseSolver):
    def __init__(self, cfg):
        super(Testsolver, self).__init__(cfg)
        
        net_name = self.cfg['model']
        lib = importlib.import_module('models.' + net_name)
        net = lib.Network
        
        # self.model = net(1, 8, args = self.cfg)
        self.model = net(args=self.cfg)  # MMALPNet

    def check(self):
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
                if gid >=0:
                    self.gpu_ids.append(gid)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_ids)
            self.model_path = os.path.join(self.cfg['pretrain']['pre_folder'], self.cfg['model'], self.cfg['satellite'], self.cfg['pretrain']['pre_source'])
            if len(self.gpu_ids) > 2:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
                device_ids = [0, 1]
                self.model = torch.nn.DataParallel(self.model, device_ids=device_ids).cuda()
                # self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
            else:
                # torch.cuda.set_device(self.gpu_ids[0])
                # self.model = self.model.cuda(self.gpu_ids[0])
                os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
                device_ids = [0, 1]
                self.model = torch.nn.DataParallel(self.model, device_ids=device_ids).cuda()
            self.model.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage)['net'])  # , map_location=lambda storage, loc: storage)['net']

    def test4refer(self):
        # self.checkpoint()
        self.model.eval()
        avg_time= []
        ssim_list = []
        rmse_list = []
        sam_list = []
        cc_list = []
        scc_list = []
        psnr_list = []
        ergas_list = []
        uiqc_list = []
        for i, batch in enumerate(self.data_loader, 0):
            gt, lms, ms, pan = Variable(batch[0], requires_grad=False), \
                    Variable(batch[1]), \
                    Variable(batch[2]), \
                    Variable(batch[3])

            if self.cuda:
                gt, lms, ms, pan = Variable(batch[0], requires_grad=False).cuda(), \
                    Variable(batch[1]).cuda(), \
                    Variable(batch[2]).cuda(), \
                    Variable(batch[3]).cuda()

            t0 = time.time()
            with torch.no_grad():
                # out1, out2, out3 = self.model(ms, pan)  # MUCNN
                # out1, out2, out3, ms_u4 = self.model(ms.float(), pan.float())  # MMAPP
                out3 = self.model(pan.float(), lms.float())  # CF2N
            t1 = time.time()
            if self.cfg['test']['use_metric']:
                ground_truth = torch.squeeze(gt,dim=0).permute(1, 2, 0).cpu().detach().numpy()
                predict_out = torch.squeeze(out3, dim=0).permute(1, 2, 0).cpu().detach().numpy()
                ssim = SSIM_numpy(ground_truth, predict_out, data_range=1)
                rmse = RMSE_numpy(ground_truth, predict_out)
                sam = SAM_numpy(ground_truth, predict_out)
                cc = CC_numpy(ground_truth, predict_out)
                scc = SCC_numpy(ground_truth, predict_out)
                psnr = MPSNR_numpy(ground_truth, predict_out, data_range=1)
                ergas = ERGAS_numpy(ground_truth, predict_out)
                uiqc = UIQC_numpy(ground_truth, predict_out)

                ssim_list.append(ssim)
                rmse_list.append(rmse)
                sam_list.append(sam)
                cc_list.append(cc)
                scc_list.append(scc)
                psnr_list.append(psnr)
                ergas_list.append(ergas)
                uiqc_list.append(uiqc)
            
            if self.cfg['test']['result2img']:
                out_name = str(self.cfg['model'].lower()) + '_' + str(self.cfg['satellite']) + '_' + str(self.cfg['test']['type']) + '-' + str(i)
                out_path2 = os.path.join('results', self.cfg['model'], self.cfg['satellite'], self.cfg['test']['type'])
                if not os.path.exists(out_path2):
                    os.makedirs(out_path2)
                out_path = os.path.join('results', self.cfg['model'], self.cfg['satellite'], self.cfg['test']['type'], str(out_name))
                if self.cfg['test']['img_type'] == 'tif':
                    # self.save_img(bms_image.cpu().data, name[0][0:-4]+'_bic.tif', mode='CMYK')
                    # self.save_img(ms_image.cpu().data, name[0][0:-4]+'_gt.tif', mode='CMYK')
                    self.save_img(out3.cpu().data, out_name +'.tif', mode='CMYK')
                    # self.save_img(pan_image.cpu().data, name[0][0:-4]+'_pan.tif', mode='L')
                elif self.cfg['test']['img_type'] == 'mat':
                    self.save_mat(out3, out_path + '.mat')
            
            print("===> Processing: %s || Timer: %.4f sec." % (i, (t1 - t0)))
            avg_time.append(t1 - t0)
        print("===> AVG Timer: %.4f sec." % (np.mean(avg_time)))

        if self.cfg['test']['use_metric']:
            avg_ssim = np.array(ssim_list).mean()
            avg_rmse = np.array(rmse_list).mean()
            avg_sam = np.array(sam_list).mean()
            avg_cc = np.array(cc_list).mean()
            avg_scc = np.array(scc_list).mean()
            avg_psnr = np.array(psnr_list).mean()
            avg_ergas = np.array(ergas_list).mean()
            avg_uiqc = np.array(uiqc_list).mean()
            print('===> Score on metrics')
            print('     AVG of SSIM: {:.4f}+-{:.4f}'.format(avg_ssim, np.array(ssim_list).std()))
            print('     AVG of PSNR: {:.4f}+-{:.4f}'.format(avg_psnr, np.array(psnr_list).std()))
            print('     AVG of SAM: {:.4f}+-{:.4f}'.format(avg_sam, np.array(sam_list).std()))
            print('     AVG of SCC: {:.4f}+-{:.4f}'.format(avg_cc, np.array(cc_list).std()))
            print('     AVG of CC: {:.4f}+-{:.4f}'.format(avg_scc, np.array(scc_list).std()))
            print('     AVG of RMSE: {:.4f}+-{:.4f}'.format(avg_rmse, np.array(rmse_list).std()))
            print('     AVG of ERGAS: {:.4f}+-{:.4f}'.format(avg_ergas, np.array(ergas_list).std()))
            print('     AVG of UIQC: {:.4f}+-{:.4f}'.format(avg_uiqc, np.array(uiqc_list).std()))
        
        
    def test4nonrefer(self):

        self.model.eval()
        avg_time = []
        d_lambda_list = []
        d_s_list = []
        qnr_list = []
        for i, batch in enumerate(self.data_loader, 0):
            lms, ms, pan = Variable(batch[0], requires_grad=False), \
                    Variable(batch[1]), \
                    Variable(batch[2])

            if self.cuda:
                lms, ms, pan = Variable(batch[0], requires_grad=False).cuda(), \
                    Variable(batch[1]).cuda(), \
                    Variable(batch[2]).cuda()

            t0 = time.time()
            with torch.no_grad():
                 # out1, out2, out3 = self.model(ms, pan)  # MUCNN
                 # out1, out2, out3, ms_u4 = self.model(ms.float(), pan.float())  # MMAPP
                out3 = self.model(pan.float(), lms.float())  # CF2N
            t1 = time.time()
            if self.cfg['test']['use_metric']:
                test_lms = torch.squeeze(lms, dim=0).permute(1, 2, 0).cpu().detach().numpy()
                test_ms = torch.squeeze(ms, dim=0).permute(1, 2, 0).cpu().detach().numpy()
                test_pan = torch.squeeze(pan, dim=0).permute(1, 2, 0).cpu().detach().numpy()
                # test_lms = lms[i, :, :, :].transpose((1, 2, 0))
                # test_ms = ms[i, :, :, :].transpose((1, 2, 0))
                # test_pan = pan[i, :, :, :].transpose((1, 2, 0)).squeeze(2)
                predict_out = torch.squeeze(out3, dim=0).permute(1, 2, 0).cpu().detach().numpy()
                d_lambda = D_lambda_numpy(test_lms, predict_out)
                d_s = D_s_numpy(test_ms, test_pan, predict_out)
                qnr = (1 - d_lambda) * (1 - d_s)

                d_lambda_list.append(d_lambda)
                d_s_list.append(d_s)
                qnr_list.append(qnr)
            
            if self.cfg['test']['result2img']:
                out_name = str(self.cfg['model'].lower()) + '_' + str(self.cfg['satellite']) + '_' + str(
                    self.cfg['test']['type']) + '-' + str(i)
                out_path2 = os.path.join('results', self.cfg['model'], self.cfg['satellite'], self.cfg['test']['type'])
                if not os.path.exists(out_path2):
                    os.makedirs(out_path2)
                out_path = os.path.join('results', self.cfg['model'], self.cfg['satellite'], self.cfg['test']['type'],
                                        str(out_name))
                if self.cfg['test']['img_type'] == 'tif':
                    # self.save_img(bms_image.cpu().data, name[0][0:-4]+'_bic.tif', mode='CMYK')
                    # self.save_img(ms_image.cpu().data, name[0][0:-4]+'_gt.tif', mode='CMYK')
                    self.save_img(out3.cpu().data, out_name + '.tif', mode='CMYK')
                    # self.save_img(pan_image.cpu().data, name[0][0:-4]+'_pan.tif', mode='L')
                elif self.cfg['test']['img_type'] == 'mat':
                    self.save_mat(out3, out_path+'.mat')

            print("===> Processing: %s || Timer: %.4f sec." % (i, (t1 - t0)))
            avg_time.append(t1 - t0)
        print("===> AVG Timer: %.4f sec." % (np.mean(avg_time)))
        if self.cfg['test']['use_metric']:
            avg_d_lambda = np.array(d_lambda_list).mean()
            avg_d_s = np.array(d_s_list).mean()
            avg_qnr = np.array(qnr_list).mean()
            print('===> Score on metrics')
            print('     AVG of D_lambda: {:.4f}+-{:.4f}'.format(avg_d_lambda, np.array(d_lambda_list).std()))
            print('     AVG of D_s: {:.4f}+-{:.4f}'.format(avg_d_s, np.array(d_s).std()))
            print('     AVG of QNR: {:.4f}+-{:.4f}'.format(avg_qnr, np.array(qnr_list).std()))

    def save_img(self, img, img_name, mode):
        save_img = img.squeeze().clamp(0, 1).numpy().transpose((1, 2, 0))
        # save img
        # img_name = str(self.cfg['model']) + ''
        save_dir=os.path.join('results', self.cfg['model'], self.cfg['satellite'], self.cfg['test']['type'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_fn = save_dir + '/' + img_name
        save_img = np.uint8(save_img*2047).astype('uint8')
        save_img = Image.fromarray(save_img, mode)
        save_img.save(save_fn)

    def save_mat(self, output, out_name):
        sr = torch.squeeze(output).permute(1, 2, 0).cpu().detach().numpy()  # HWC
        sio.savemat(out_name, {'sr': sr})
  
    def run(self):
        self.check()
        if self.cfg['test']['type'] == 'rr':            
            self.dataset = Load_RRTset(self.cfg['test']['test_set'] + '/' + self.cfg['test']['test4refer'])
            self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=1, num_workers=self.cfg['num_worker'])
            self.test4refer()
        elif self.cfg['test']['type'] == 'fr':            
            self.dataset = Load_FRTset(self.cfg['test']['test_set'] + '/' + self.cfg['test']['test4nonrefer'])
            self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=1, num_workers=self.cfg['num_worker'])
            self.test4nonrefer()
        else:
            raise ValueError('Mode error!')