# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 23:04:11 2022
@author: likun
"""

import os
import time
import cv2
import numpy as np
import scipy.ndimage
import scipy.io as sio  

import h5py 
import torch                                
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import ref_evaluate, no_ref_evaluate

from argparse import ArgumentParser
parser = ArgumentParser(description='Trainable-Nets')
parser.add_argument('--Network', type=str,   default='pro_fusionnet', help='from {fusionnet_v1,2,...  }')
parser.add_argument('--Dataset', type=str,   default='CAVE',  help='training dataset {CAVE, Harvard, Chikusei_v2, GF5 }')
parser.add_argument('--testing_epoch', type=int,   default=300, help='select a specific epoch number of trained models') 

parser.add_argument('--gpu_list', type=str,   default='0', help='gpu index') 
parser.add_argument('--trained_epoch', type=int,   default=301, help='epoch number of end training')
parser.add_argument('--model_dir', type=str,   default='train_models', help='trained or pre-trained model directory') 
parser.add_argument('--fusion_test_dir', type=str,   default='fusion_tests', help='fusion test of the pre-trained model')
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
import h5py 
ref,hsi,msi,pan  = [],[],[],[]

#################### CAVE #######################  
if args.Dataset =='CAVE':
    ms_path  = './data/CAVE/version_1/test'
    # ms_path  = 'E:\\0 HyperModel\\1_dealed_data\\CAVE\\version_1\\test'                                        
    ms_file_list = os.listdir(ms_path)  
    ms_file_list.sort(key=lambda x:int(x.split('_simu')[0]))                               
    for file in ms_file_list:                                                                 
        if not os.path.isdir(file):                                                             
            mat_data = h5py.File(ms_path+"/"+file)     
            refs   = np.array(mat_data["I_ref"],dtype ='float32').T
            ref.append(refs)
            hsis   = np.array(mat_data["I_H"],dtype ='float32').T 
            hsi.append(hsis)   
            msis   = np.array(mat_data["I_M"],dtype ='float32').T  
            msi.append(msis)   
            pans   = np.array(mat_data["I_P"],dtype ='float32').T    
            pans   = np.expand_dims(pans,-1)          
            pan.append(pans)  
    print('ref.len' ,len(ref)) 

#################### Harvard #######################  
elif args.Dataset =='Harvard':
    print("Set you customed training dataset")
else:
    raise NotImplementedError("Dataset [%s] is not recognized." % args.Dataset)



from model.pro_fusionnet import fusionnet as net
MC = msis.shape[2]  
HC = hsis.shape[2] 
lamda_m, lamda_p = 0.01, 0.01
lamda_m = torch.tensor(lamda_m).float().view([1, 1, 1, 1])
lamda_p = torch.tensor(lamda_p).float().view([1, 1, 1, 1])  
[lamda_m, lamda_p] = [el.to(device) for el in [lamda_m, lamda_p]]
n_iter = 5
    
if args.Dataset in ['CAVE', 'Harvard', 'Chikusei_v2']:  
    model = net(n_iter =n_iter, h_nc = 64, in_c= HC+1, out_c= HC, m_c= MC, nc=[80, 160, 320],  
                                nb   = 1,  act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")            
elif args.Dataset =='GF5_real':  
    model = net(n_iter =n_iter, h_nc = 64, in_c= HC+1, out_c= HC, m_c= MC, nc=[280, 420, 560],  
                                nb   = 1,  act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
else:
    raise NotImplementedError("Network [%s] is not recognized." % args.Network)

# Load pre-trained model with epoch number  
model       = nn.DataParallel(model).to(device)    
model_files = "Net_%s_Data_%s_%d_Epoch_%d_" %(args.Network, args.Dataset, n_iter, args.trained_epoch)   
model.load_state_dict(torch.load('./%s/%s/net_params_%d.pkl' % (args.model_dir, model_files, args.testing_epoch)))
print   ("Test_Network is:",model,'\n')
print   ("Model_File_Name is:\n",model_files,'\n')
 

with torch.no_grad():
    for img_id in range(len(ref)):
        test_ref  =  ref[img_id]         
        test_hsi  =  hsi[img_id]
        test_msi  =  msi[img_id]      
        test_pan  =  pan[img_id]
        batch_R =  np.expand_dims(test_ref, 0)
        batch_H =  np.expand_dims(test_hsi, 0)
        batch_M =  np.expand_dims(test_msi, 0) 
        batch_P =  np.expand_dims(test_pan, 0)
        batch_r = ((torch.tensor(batch_R).permute(0, 3, 1, 2)).type(torch.FloatTensor)).to(device) 
        batch_h = ((torch.tensor(batch_H).permute(0, 3, 1, 2)).type(torch.FloatTensor)).to(device)  
        batch_m = ((torch.tensor(batch_M).permute(0, 3, 1, 2)).type(torch.FloatTensor)).to(device) 
        batch_p = ((torch.tensor(batch_P).permute(0, 3, 1, 2)).type(torch.FloatTensor)).to(device) 
        start   = time.time() 
        if args.Network == 'pro_fusionnet': 
            batch_pred = model(batch_h, batch_m, batch_p, lamda_m, lamda_p) 
        elif args.Network == 'GSA_SSR':
            batch_pred, _, _,  _, _, _ = model(batch_H, batch_M, batch_P)
        else:
            raise NotImplementedError("Network [%s] model error occurs." % args.Network)
        end  = time.time()
        tim  = end - start    
        batch_pred_np = batch_pred.permute(0, 2, 3, 1).cpu().data.numpy() 
        batch_pred_np = np.clip(batch_pred_np, 0, 1)  
        test_label    = batch_pred_np[0,:,:,:]
        if (img_id==len(ref)-1):
            print('test_label',test_label.shape)
        save_fusion_img_dir ="./%s/" %(args.fusion_test_dir) + model_files + "Testepoch_%d/" %(args.testing_epoch)
        save_fusion_mat_dir ="./%s/" %(args.fusion_test_dir) + model_files + "Testepoch_%d/" %(args.testing_epoch)
        save_fusion_log_dir ="./%s/" %(args.fusion_test_dir) + model_files + "Testepoch_%d/" %(args.testing_epoch)
        if not os.path.exists(save_fusion_img_dir):
            os.makedirs(save_fusion_img_dir)
        if not os.path.exists(save_fusion_mat_dir):
            os.makedirs(save_fusion_mat_dir)
        if not os.path.exists(save_fusion_log_dir):
            os.makedirs(save_fusion_log_dir)
        if args.Dataset =='Chikusei_v2':  
            cv2.imwrite (save_fusion_img_dir +'%d_test.png'%(img_id+1) ,np.uint8(255*test_label)[:, :, [10,20,30]] )
            cv2.imwrite (save_fusion_img_dir +'%d_ref.png'%(img_id+1)  ,np.uint8(255*test_ref)  [:, :, [10,20,30]] )
        elif args.Dataset == 'CAVE':
            cv2.imwrite (save_fusion_img_dir +'%d_test.png'%(img_id+1) ,np.uint8(255*test_label)[:, :, [5,15,25]] )  
            cv2.imwrite (save_fusion_img_dir +'%d_ref.png'%(img_id+1)  ,np.uint8(255*test_ref)  [:, :, [5,15,25]] )
        elif args.Dataset == 'GF5_real':
            test_ref = scipy.ndimage.zoom(test_ref, (15,15,1), order=0)
            cv2.imwrite (save_fusion_img_dir +'%d_test.png'%(img_id+1) ,np.uint8(255*test_label)[:, :, [50,110,170]] )       
            cv2.imwrite (save_fusion_img_dir +'%d_exp.png'%(img_id+1)  ,np.uint8(255*test_ref)  [:, :, [50,110,170]] )
        # save 0-1 mat
        if args.Dataset == 'GF5_real':
            sio.savemat (save_fusion_mat_dir  +'Norm_%s_%d.mat'%(args.Network, img_id+1),
                         {'fusion':test_label,'hsi':test_hsi,'msi':test_msi,'pan':test_pan} )
        else:
            sio.savemat (save_fusion_mat_dir  +'Norm_%s_%d.mat'%(args.Network, img_id+1),
                         {'ref':test_ref, 'fusion':test_label} )

        ref_results={}
        ref_results.update({'metrics: ':'  PSNR,   SSIM,   SAM,   ERGAS,  SCC,    Q,    RMSE'})
        no_ref_results={}
        no_ref_results.update({'metrics: ':'  D_lamda, D_s,    QNR'})
        temp_ref_results      = ref_evaluate( np.uint8(255*test_label), np.uint8(255*test_ref) )  
        # temp_no_ref_results = no_ref_evaluate( test_label,  LR_pan ,  LR_ms )  
        ref_results   .update({'xxx     ':temp_ref_results})
        # no_ref_results.update({'xxx     ':temp_no_ref_results})
        pro_fusionnetoutput_file_ref    = save_fusion_log_dir+"%s_Dataset_ref.txt" % (args.Dataset)    
        # pro_fusionnetoutput_file_no_ref = save_fusion_log_dir+"%s_Dataset_no_ref.txt" % (args.Dataset)     
        
        print('################## reference  #######################')
        for index, i in enumerate(ref_results):
            if index == 0:
                print(i, ref_results[i])
        else:    
                print(i, [round(j, 4) for j in ref_results[i]])
                list2str= str([ round(j, 4) for j in ref_results[i] ])
                list2str= ('%d  '+ list2str+'\n')%(img_id+1) 
                pro_fusionnetoutput_file = open(pro_fusionnetoutput_file_ref, 'a')
                pro_fusionnetoutput_file.write(list2str)
                pro_fusionnetoutput_file.close()  
        # print('################## no reference  ####################')
        # for index, i in enumerate(no_ref_results):
        #      if index == 0:9
        #          print(i, no_ref_results[i])
        #      else:    
        #          print(i, [round(j, 4) for j in no_ref_results[i]])
        #          list2str= str([ round(j, 4) for j in no_ref_results[i] ])     
        #          list2str=('%d  '+ list2str+'\n')%(img_id+1) 
        #          pro_fusionnetoutput_file = open(pro_fusionnetoutput_file_no_ref, 'a')
        #          pro_fusionnetoutput_file.write(list2str)
        #          pro_fusionnetoutput_file.close()  
        # print('#####################################################')
      
    print('test finished')
