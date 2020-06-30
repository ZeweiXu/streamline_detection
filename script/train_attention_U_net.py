# Load all the dependencies
import os
import sys
import random
import warnings
import numpy as np
from numpy import genfromtxt
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
np.random.seed(1337) # for reproducibility
from tensorflow import set_random_seed
set_random_seed(1337)
from itertools import chain
from keras.layers import Layer,UpSampling2D,merge
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers import Reshape, Permute, Input, add, multiply
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from keras.layers import dot
# To specify the GPU ID uncomment this block and  
# with K.tf.device('/gpu:0'): # specify the ID of GPU here (0: the first GPU)
#    config = tf.ConfigProto(intra_op_parallelism_threads=4,\
#           inter_op_parallelism_threads=4, allow_soft_placement=True,\
#           device_count = {'CPU' : 1, 'GPU' : 1})
#    session = tf.Session(config=config)
#    K.set_session(session)

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# Use dice coefficient function as the loss function 
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

# Jacard coefficient
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

# calculate loss value
def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

# calculate loss value
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def Residual_CNN_block(x, size, dropout=0.0, batch_norm=True):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    return conv

class multiplication(Layer):
    def __init__(self,inter_channel = None,**kwargs):
        super(multiplication, self).__init__(**kwargs)
        self.inter_channel = inter_channel
    def build(self,input_shape=None):
        self.k = self.add_weight(name='k',shape=(1,),initializer='zeros',dtype='float32',trainable=True)
    def get_config(self):
        base_config = super(multiplication, self).get_config()
        config = {'inter_channel':self.inter_channel}
        return dict(list(base_config.items()) + list(config.items()))  
    def call(self,inputs):
        g,x,x_query,phi_g,x_value = inputs[0],inputs[1],inputs[2],inputs[3],inputs[4]
        h,w,c = int(x.shape[1]),int(x.shape[2]),int(x.shape[3])
        x_query = K.reshape(x_query, shape=(-1,h*w, self.inter_channel//4))
        phi_g = K.reshape(phi_g,shape=(-1,h*w,self.inter_channel//4))
        x_value = K.reshape(x_value,shape=(-1,h*w,c))
        scale = dot([K.permute_dimensions(phi_g,(0,2,1)), x_query], axes=(1, 2))
        soft_scale = Activation('softmax')(scale)
        scaled_value = dot([K.permute_dimensions(soft_scale,(0,2,1)),K.permute_dimensions(x_value,(0,2,1))],axes=(1, 2))
        scaled_value = K.reshape(scaled_value, shape=(-1,h,w,c))        
        customize_multi = self.k * scaled_value
        layero = add([customize_multi,x])
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
        concate = my_concat([layero,g])
        return concate 
    def compute_output_shape(self,input_shape):
        ll = list(input_shape)[1]
        return (None,ll[1],ll[1],ll[3]*3)
    def get_custom_objects():
        return {'multiplication': multiplication}

def attention_up_and_concatenate(inputs):
    g,x = inputs[0],inputs[1]
    inter_channel = g.get_shape().as_list()[3]
    g = Conv2DTranspose(inter_channel, (2,2), strides=[2, 2],padding='same')(g)
    x_query = Conv2D(inter_channel//4, [1, 1], strides=[1, 1], data_format='channels_last')(x)
    phi_g = Conv2D(inter_channel//4, [1, 1], strides=[1, 1], data_format='channels_last')(g)
    x_value = Conv2D(inter_channel//2, [1, 1], strides=[1, 1], data_format='channels_last')(x)
    inputs = [g,x,x_query,phi_g,x_value]
    concate = multiplication(inter_channel)(inputs)
    return concate

class multiplication2(Layer):
    def __init__(self,inter_channel = None,**kwargs):
        super(multiplication2, self).__init__(**kwargs)
        self.inter_channel = inter_channel
    def build(self,input_shape=None):
        self.k = self.add_weight(name='k',shape=(1,),initializer='zeros',dtype='float32',trainable=True)
    def get_config(self):
        base_config = super(multiplication2, self).get_config()
        config = {'inter_channel':self.inter_channel}
        return dict(list(base_config.items()) + list(config.items()))  
    def call(self,inputs):
        g,x,rate = inputs[0],inputs[1],inputs[2]
        scaled_value = multiply([x, rate])
        att_x =  self.k * scaled_value
        att_x = add([att_x,x])
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
        concate = my_concat([att_x, g])
        return concate 
    def compute_output_shape(self,input_shape):
        ll = list(input_shape)[1]
        return (None,ll[1],ll[1],ll[3]*2)
    def get_custom_objects():
        return {'multiplication2': multiplication2}

def attention_up_and_concatenate2(inputs):
    g, x = inputs[0],inputs[1]
    inter_channel = g.get_shape().as_list()[3]
    g = Conv2DTranspose(inter_channel//2, (3,3), strides=[2, 2],padding='same')(g)
    g = Conv2D(inter_channel//2, [1, 1], strides=[1, 1], data_format='channels_last')(g)
    theta_x = Conv2D(inter_channel//4, [1, 1], strides=[1, 1], data_format='channels_last')(x)
    phi_g = Conv2D(inter_channel//4, [1, 1], strides=[1, 1], data_format='channels_last')(g)
    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format='channels_last')(f)
    rate = Activation('sigmoid')(psi_f)
    concate =  multiplication2()([g,x,rate])
    return concate

def UNET_224(weights=None):
    inputs = Input((IMG_WIDTH, IMG_WIDTH, INPUT_CHANNELS))
    filters = 32
    last_dropout = 0.2
# convolutiona and pooling level 1
    conv_224 = Residual_CNN_block(inputs,filters)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)
# convolutiona and pooling level 2
    conv_112 = Residual_CNN_block(pool_112,2*filters)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)
# convolutiona and pooling level 3
    conv_56 = Residual_CNN_block(pool_56,4*filters)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)
# convolutiona and pooling level 4
    conv_28 = Residual_CNN_block(pool_28,8*filters)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)
# convolutiona and pooling level 5
    conv_14 = Residual_CNN_block(pool_14,16*filters)
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)
# Conlovlution and feature concatenation
    conv_7 = Residual_CNN_block(pool_7,32*filters)
# Upsampling with convolution 
    up_14 = attention_up_and_concatenate([conv_7, conv_14]) 
    up_conv_14 = Residual_CNN_block(up_14,16*filters)
# Upsampling with convolution 2
    up_28 = attention_up_and_concatenate([up_conv_14, conv_28])
    up_conv_28 = Residual_CNN_block(up_28,8*filters)
# Upsampling with convolution 3
    up_56 = attention_up_and_concatenate2([up_conv_28, conv_56])
    up_conv_56 = Residual_CNN_block(up_56,4*filters)
# Upsampling with convolution 4
    up_112 = attention_up_and_concatenate2([up_conv_56, conv_112])
    up_conv_112 = Residual_CNN_block(up_112,2*filters)
# Upsampling with convolution 5
    up_224 = attention_up_and_concatenate2([up_conv_112, conv_224])
    #up_224 = attention_up_and_concatenate2(up_conv_112, conv_224)
    up_conv_224 = Residual_CNN_block(up_224,filters,dropout = last_dropout)
# 1 dimensional convolution and generate probabilities from Sigmoid function
    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_224)  
    conv_final = Activation('sigmoid')(conv_final)
# Generate model
    model = Model(inputs, conv_final, name="UNET_224")
    return model

import numpy as np
scenario = raw_input("Please specify the scenario (up,down,left,or right):")
aug = '_aug_'+scenario 
# read in training and validation data
X_train = np.load('../data/train_data'+aug+'.npy')#[:2000]
Y_train = np.load('../data/train_label'+aug+'.npy')#[:,:,:,np.newaxis]#[:,:,:,np.newaxis]#[:2000]
X_Validation = np.load('../data/vali_data'+aug.split('_aug')[-1]+'.npy')#[:700]
Y_Validation = np.load('../data/vali_label'+aug.split('_aug')[-1]+'.npy')
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

print(X_train.shape)
print(Y_train.shape)
print(X_Validation.shape)
print(Y_Validation.shape)

patch_size = 224

IMG_WIDTH = patch_size
IMG_HEIGHT = patch_size
# Number of feature channels 
INPUT_CHANNELS = 8
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1
maxepoch = 300
# hyperparameters
learning_rate =0.0000359
patience = 20

# create the CNN
model = UNET_224()
model.compile(optimizer=Adam(lr=learning_rate),loss = dice_coef_loss,metrics=[dice_coef,'accuracy'])
callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=patience, min_lr=1e-9, verbose=1, mode='min'),
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint('model'+aug+'_attention2.h5', monitor='val_loss', save_best_only=True, verbose=0),
    ]

results_03 = model.fit(X_train, Y_train, validation_data=(X_Validation,Y_Validation), batch_size=16, epochs=maxepoch, callbacks=callbacks)
from keras.models import load_model
import pickle
# save the intermdediate results and training statistics
with open('../models/history'+aug+'_attention_U_net.pickle', 'wb') as file_pi:
    pickle.dump(results_03.history, file_pi, protocol=2)
# save the model
model.save('../models/model'+aug+'_attention_U_net.h5')
# load weights into new model
newmodel = load_model('../models/model'+aug+'_attention_U_net.h5', custom_objects={'multiplication': multiplication,'multiplication2': multiplication2,"dice_coef_loss":dice_coef_loss, "dice_coef":dice_coef})
# Reload the model and save the predicted labels.
X_test = np.load("../data/prediction_data.npy")
preds_test = newmodel.predict(X_test)
preds_test_t = (preds_test > 0.5).astype(np.uint8)
np.save('../result/preds_test_total'+aug+'_attention_U_net.npy',preds_test_t)

