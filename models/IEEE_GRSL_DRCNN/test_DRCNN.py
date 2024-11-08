###########################################################################
# Created by: Yuxuan Zheng
# Email: yxzheng24@163.com
# Testing code for paper titled "Deep Residual Learning for Boosting the Accuracy of Hyperspectral Pansharpening"

# Citation
# Y. Zheng, J. Li, Y. Li, K. Cao and K. Wang, "Deep Residual Learning for Boosting the Accuracy of Hyperspectral Pansharpening," 
# in IEEE Geoscience and Remote Sensing Letters, vol. 17, no. 8, pp. 1435-1439, Aug. 2020, doi: 10.1109/LGRS.2019.2945424.
###########################################################################

from __future__ import absolute_import, division

import numpy as np
from keras.models import Model
import h5py
import scipy.io as sio

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from train_DRCNN import eval_drcnn

if __name__ == "__main__":
      
    inputs, outputs = eval_drcnn()
    model = Model(inputs=inputs, outputs=outputs)
    model.load_weights('./models/model_cave.h5', by_name=True)
    
    for i in range(10):
        
        ind = i+1
        
        print ('processing for %d'%ind)

        data = h5py.File('./data_process/cave_Hini/test_10Hini/%d.mat'%ind)
        data = np.transpose(data['I_gf'])
        
        data = np.expand_dims(data, 0)
    
        data_res = model.predict(data, batch_size=1, verbose=1)
        
        data_res = np.reshape(data_res, (512, 512, 31))
        
        data_fus = np.squeeze(data, axis = 0) + data_res
        
        data_fus = np.array(data_fus, dtype=np.float64)
        
        sio.savemat('./get_10Hfus/getHfus_%d.mat'%ind, {'Hfus': data_fus}) 
  