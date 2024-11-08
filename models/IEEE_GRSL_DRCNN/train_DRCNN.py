###########################################################################
# Created by: Yuxuan Zheng
# Email: yxzheng24@163.com
# Training code for paper titled "Deep Residual Learning for Boosting the Accuracy of Hyperspectral Pansharpening"

# Citation
# Y. Zheng, J. Li, Y. Li, K. Cao and K. Wang, "Deep Residual Learning for Boosting the Accuracy of Hyperspectral Pansharpening," 
# IEEE Geoscience and Remote Sensing Letters, vol. 17, no. 8, pp. 1435-1439, Aug. 2020, doi: 10.1109/LGRS.2019.2945424.
###########################################################################

from __future__ import absolute_import, division
from keras.layers import Input, Conv2D, Activation, BatchNormalization

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot
import h5py
from keras.callbacks import ModelCheckpoint

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

def tf_log10(x):
    n = tf.log(x)
    d = tf.log(tf.constant(10, dtype = n.dtype))
    return n/d

def psnr(y_ture, y_pred):
    max_pixel =1.0
    return 10.0*tf_log10((max_pixel**2)/(K.mean(K.square(y_pred-y_ture))))
    
def read_data(path):
  """
  Read h5 format data file
  
  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def eval_drcnn():
    inputs = l = Input((512, 512, 31), name='input')

    # conv11
    l = Conv2D(64, (3, 3), padding='same', name='conv11')(l)
    l = Activation('relu', name='conv11_relu')(l)

    # conv12
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv12')(l)
    l = BatchNormalization(name='conv12_bn')(l)
    l = Activation('relu', name='conv12_relu')(l)

    # conv13
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv13')(l)
    l = BatchNormalization(name='conv13_bn')(l)
    l = Activation('relu', name='conv13_relu')(l)

    # conv14
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv14')(l)
    l = BatchNormalization(name='conv14_bn')(l)
    l = Activation('relu', name='conv14relu')(l)

    #conv15
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv15')(l)
    l = BatchNormalization(name='conv15_bn')(l)
    l = Activation('relu', name='conv15_relu')(l)
    
    #conv16
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv16')(l)
    l = BatchNormalization(name='conv16_bn')(l)
    l = Activation('relu', name='conv16_relu')(l)
    
    #conv17    
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv17')(l)
    l = BatchNormalization(name='conv17_bn')(l)
    l = Activation('relu', name='conv17_relu')(l)
    
    #conv18
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv18')(l)
    l = BatchNormalization(name='conv18_bn')(l)
    l = Activation('relu', name='conv18_relu')(l)
    
    #conv19
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv19')(l)
    l = BatchNormalization(name='conv19_bn')(l)
    l = Activation('relu', name='conv19_relu')(l)
    
    #conv20
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv20')(l)
    l = BatchNormalization(name='conv20_bn')(l)
    l = Activation('relu', name='conv20_relu')(l)
    
    #conv21
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv21')(l)
    l = BatchNormalization(name='conv21_bn')(l)
    l = Activation('relu', name='conv21_relu')(l)
    
    #conv22
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv22')(l)
    l = BatchNormalization(name='conv22_bn')(l)
    l = Activation('relu', name='conv22_relu')(l)

    #conv23
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv23')(l)
    l = BatchNormalization(name='conv23_bn')(l)
    l = Activation('relu', name='conv23_relu')(l)
    
    #conv24
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv24')(l)
    l = BatchNormalization(name='conv24_bn')(l)
    l = Activation('relu', name='conv24_relu')(l)
    
   #conv25
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv25')(l)
    l = BatchNormalization(name='conv25_bn')(l)
    l = Activation('relu', name='conv25_relu')(l)
    
    l = Conv2D(31, (3, 3), padding='same', strides=(1, 1), name='conv30')(l)
        
    # out
    outputs = l

    return inputs, outputs


def train_drcnn():
    inputs = l = Input((32, 32, 31), name='input')

    # conv11
    l = Conv2D(64, (3, 3), padding='same', name='conv11')(l)
    l = Activation('relu', name='conv11_relu')(l)

    # conv12
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv12')(l)
    l = BatchNormalization(name='conv12_bn')(l)
    l = Activation('relu', name='conv12_relu')(l)

    # conv13
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv13')(l)
    l = BatchNormalization(name='conv13_bn')(l)
    l = Activation('relu', name='conv13_relu')(l)

    # conv14
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv14')(l)
    l = BatchNormalization(name='conv14_bn')(l)
    l = Activation('relu', name='conv14relu')(l)

    #conv15
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv15')(l)
    l = BatchNormalization(name='conv15_bn')(l)
    l = Activation('relu', name='conv15_relu')(l)
    
    #conv16
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv16')(l)
    l = BatchNormalization(name='conv16_bn')(l)
    l = Activation('relu', name='conv16_relu')(l)
    
    #conv17    
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv17')(l)
    l = BatchNormalization(name='conv17_bn')(l)
    l = Activation('relu', name='conv17_relu')(l)
    
    #conv18
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv18')(l)
    l = BatchNormalization(name='conv18_bn')(l)
    l = Activation('relu', name='conv18_relu')(l)
    
    #conv19
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv19')(l)
    l = BatchNormalization(name='conv19_bn')(l)
    l = Activation('relu', name='conv19_relu')(l)
    
    #conv20
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv20')(l)
    l = BatchNormalization(name='conv20_bn')(l)
    l = Activation('relu', name='conv20_relu')(l)
    
    #conv21
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv21')(l)
    l = BatchNormalization(name='conv21_bn')(l)
    l = Activation('relu', name='conv21_relu')(l)
    
    #conv22
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv22')(l)
    l = BatchNormalization(name='conv22_bn')(l)
    l = Activation('relu', name='conv22_relu')(l)

    #conv23
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv23')(l)
    l = BatchNormalization(name='conv23_bn')(l)
    l = Activation('relu', name='conv23_relu')(l)
    
    #conv24
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv24')(l)
    l = BatchNormalization(name='conv24_bn')(l)
    l = Activation('relu', name='conv24_relu')(l)
        
   #conv25
    l = Conv2D(64, (3, 3), padding='same', strides=(1, 1), name='conv25')(l)
    l = BatchNormalization(name='conv25_bn')(l)
    l = Activation('relu', name='conv25_relu')(l)
    
    l = Conv2D(31, (3, 3), padding='same', strides=(1, 1), name='conv30')(l)
    
    # out
    outputs = l

    return inputs, outputs
    
if __name__ == "__main__":
    data_dir = os.path.join('./data_process/train_cave.h5')
    
    train_data, train_label = read_data(data_dir)
    
    train_data = np.transpose(train_data,(0,2,3,1))
    train_label = np.transpose(train_label,(0,2,3,1))
        
    inputs, outputs = train_drcnn()
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    optim = Adam(1e-3)
    loss = 'mse'
    model.compile(optim, loss, metrics=[psnr])
    
    checkpointer = ModelCheckpoint(filepath="./models/model_cave.h5", verbose=1, save_best_only=True)
    
    x_train = train_data
    y_train = train_label
    
    history = model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_split=0.2, initial_epoch=0, callbacks=[checkpointer])
    
    pyplot.plot(history.history['loss'], label='train_loss')
    pyplot.plot(history.history['val_loss'], label='val_loss')
    pyplot.legend()
    pyplot.show()
    