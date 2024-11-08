# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 22:13:20 2022
@author: likun
"""

import time
import torch     
import torch.nn as nn
import torch.nn.functional as F   
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
import os 
import cv2
import h5py 
import numpy as np
import scipy.io as sio  

from argparse import ArgumentParser
parser = ArgumentParser(description='Trainable-Nets')

parser.add_argument('--Network', type=str, default='pro_fusionnet', help='from {fusionnet_v1,v2...  }')
parser.add_argument('--Dataset', type=str, default='CAVE',  help='training dataset from {CAVE, Harvard, Chikusei_v2 }')
parser.add_argument('--batch_size', type=int, default =10,  help='{CAVE-10; Chikusei-8; GF5_simu-8l;  GF5_real-6}') 

parser.add_argument('--gpu_list', type=str, default ='0', help='gpu index') 
parser.add_argument('--learning_rate',type=float, default =5*1e-4, help='learning rate') 
parser.add_argument('--WEIGHT_DECAY', type=float, default =1e-8, help='params of ADAM') 
parser.add_argument('--end_epoch', type=int, default =301, help='epoch number of end training 351 ')
parser.add_argument('--start_epoch', type=int, default =0, help='epoch number of start training')
parser.add_argument('--model_dir', type=str, default ='train_models', help='trained or pre-trained model directory') 
parser.add_argument('--name', type=str, default='', help='renames results.txt to results_name.txt if supplied')
args = parser.parse_args() 
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
########## CAVEdata #################  
if args.Dataset =='CAVE':
    path      = './data/CAVE/version_1/'
    # path      = 'E:\\0 HyperModel\\1_dealed_data\\CAVE\\version_1\\'
    train_hsi = np.load(path+'CAVE_data_simu_train_cropH.npy')
    train_msi = np.load(path+'CAVE_data_simu_train_cropM.npy')
    train_pan = np.load(path+'CAVE_data_simu_train_cropP.npy')
    train_ref = np.load(path+'CAVE_data_simu_train_cropR.npy')
elif args.Dataset =='Chikusei_v2':  
    path      = 'E:\\0 HyperModel\\1_dealed_data\\Chikusei\\version_1\\'
    train_hsi = np.load(path+'v2_Chikusei_data_simu_train_cropH.npy')
    train_msi = np.load(path+'v2_Chikusei_data_simu_train_cropM.npy')
    train_pan = np.load(path+'v2_Chikusei_data_simu_train_cropP.npy')
    train_ref = np.load(path+'v2_Chikusei_data_simu_train_cropR.npy')  
elif args.Dataset =='Harvard':
    path      = 'E:\\0 HyperModel\\1_dealed_data\\Harvard\\version_1\\'
    train_hsi = np.load(path+'Harvard_data_simu_train_cropH.npy')
    train_msi = np.load(path+'Harvard_data_simu_train_cropM.npy')
    train_pan = np.load(path+'Harvard_data_simu_train_cropP.npy')
    train_ref = np.load(path+'Harvard_data_simu_train_cropR.npy')   

    raise NotImplementedError("Dataset [%s] is not recognized." % args.Dataset)


from model.pro_fusionnet import fusionnet as net
MC     = train_msi.shape[3]  
HC     = train_ref.shape[3]
patchs = train_ref.shape[0]  
n_iter = 5
if args.Dataset in ['CAVE', 'Harvard', 'Chikusei_v2']:  
    model = net(n_iter =n_iter, h_nc = 64, in_c= HC+1, out_c= HC, m_c= MC, nc=[80, 160, 320],  
                                nb   = 1,  act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
elif args.Dataset =='GF5_real':  
    model = net(n_iter =n_iter, h_nc = 64, in_c= HC+1, out_c= HC, m_c= MC, nc=[280, 420, 560],                  
                                nb   = 1,  act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
else:
    raise NotImplementedError("Network [%s] is not recognized." % args.Network)

Name      = "Net_%s_Data_%s_%d_Epoch_%d_" %(args.Network, args.Dataset, n_iter, args.end_epoch)
model_pth =  "./%s/" %(args.model_dir)+Name
log_dir   = "./%s/" %(args.model_dir)+Name+'.txt' 
if not os.path.exists(model_pth):
    os.makedirs(model_pth)    
lamda_m = 0.01  
lamda_p = 0.01  
lamda_m = torch.tensor(lamda_m).float().view([1, 1, 1, 1])
lamda_p = torch.tensor(lamda_p).float().view([1, 1, 1, 1])  
[lamda_m, lamda_p] = [el.to(device) for el in [lamda_m, lamda_p]] 

print        ("Train_Network is:",model)
model      = nn.DataParallel(model).to(device) 
optimizer  = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.WEIGHT_DECAY)
scheduler  = MultiStepLR(optimizer, milestones=[25,50,100], gamma=0.5)

class RandomDataset(Dataset):
    def __init__(self, hsi, msi, pan, ref, length):
        self.len   = length        
        self.hsi  = torch.tensor(hsi).permute(0, 3, 1, 2)
        self.msi  = torch.tensor(msi).permute(0, 3, 1, 2)
        self.pan  = torch.tensor(pan).permute(0, 3, 1, 2)
        self.ref  = torch.tensor(ref).permute(0, 3, 1, 2) 
    def __getitem__(self, index):
        return self.hsi[index].float(),self.msi[index].float(),self.pan[index].float(),self.ref[index].float()
    def __len__(self):
        return self.len
random_loader = DataLoader(dataset     =  RandomDataset(train_hsi, train_msi, train_pan, train_ref, patchs), 
                           batch_size  = args.batch_size, 
                           shuffle     = True, 
                           num_workers = 0
                           )
def gradient_diff(GT, Pred):  
    R = F.pad(GT, [0, 1, 0, 0])[:, :, :, 1:] 
    B = F.pad(GT, [0, 0, 0, 1])[:, :, 1:, :]
    dx1, dy1 = torch.abs(R - GT), torch.abs(B - GT)
    dx1[:, :, :, -1], dy1[:, :, -1, :] = 0, 0 
    R = F.pad(Pred, [0, 1, 0, 0])[:, :, :, 1:] 
    B = F.pad(Pred, [0, 0, 0, 1])[:, :, 1:, :]
    dx2, dy2 = torch.abs(R - Pred), torch.abs(B - Pred)
    dx2[:, :, :, -1], dy2[:, :, -1, :] = 0, 0   
    res = torch.abs(dx2-dx1)+torch.abs(dy2-dy1)
    return res

import skimage.measure
from torch.utils.tensorboard import SummaryWriter
print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
tb_writer = SummaryWriter(comment=args.Network+'_'+args.Dataset+args.name)
start = time.time()   
# Training loop
for epoch_i in range(args.start_epoch+1, args.end_epoch+1):
    PSNR, MSE, i = 0, 0, 0
    for data in random_loader:
        batch_h = data[0].to(device) 
        batch_m = data[1].to(device)
        batch_p = data[2].to(device)
        batch_r = data[3].to(device)

        # Model_Predict and loss_Compute
        if args.Network == 'pro_fusionnet':
            batch_pred = model(batch_h, batch_m, batch_p, lamda_m, lamda_p)
            loss       = torch.mean(torch.abs(batch_r - batch_pred))  
            loss_grad  = torch.mean(gradient_diff(batch_r, batch_pred))  
        else:
            raise NotImplementedError("Network [%s] model error occurs." % args.Network)
        loss_all = loss + 0.3*loss_grad   
        psnr     = skimage.metrics.peak_signal_noise_ratio(batch_r.cpu().numpy(), batch_pred.cpu().data.numpy(), data_range=1 )        
        mse      = skimage.metrics.mean_squared_error(batch_r.cpu().numpy(), batch_pred.cpu().data.numpy())    
        PSNR     = (PSNR*i+psnr)*1.0/(i+1)
        MSE      = (MSE*i+mse)*1.0/(i+1) 

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step() 
        scheduler.step()
    output_data = "[%02d/%02d] Total Loss: %.6f loss: %.6f  loss_grad: %.6f MSE %.6f PSNR %.3f\n" % (
        epoch_i, args.end_epoch, loss_all.item(), loss, loss_grad, MSE,PSNR
    )
    print(output_data)
    output_file = open(log_dir, 'a')
    output_file.write(output_data)
    output_file.close()
    if epoch_i % 10 == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_pth, epoch_i))

    # Write into tensorboard                                                      
    now_lr = optimizer.param_groups[0]["lr"]
    if tb_writer:
        tags = ['train/loss_all', "learning_rate", "PSNR", "MSE"]
        for x, tag in zip([loss_all, now_lr, PSNR, MSE], tags):
            tb_writer.add_scalar(tag, x, epoch_i) 

end  = time.time()
tim  = end - start  
print(tim)
curtime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
print(curtime)
 
