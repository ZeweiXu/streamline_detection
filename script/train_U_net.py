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
from keras.layers import UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf

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

# Double convolutional layer followed by batch normalization and dropout layer  that used at every horizontal level of convolution and transpose convolution
def conv_layer(x, size, dropout=0.0, batch_norm=True):
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

    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv

# The structure of the constructed U-net model
def UNET_224(dropout_val=0.2, weights=None):
    if K.image_dim_ordering() == 'th':
        inputs = Input((INPUT_CHANNELS, IMG_WIDTH, IMG_WIDTH))
        axis = 1
    else:
        inputs = Input((IMG_WIDTH, IMG_WIDTH, INPUT_CHANNELS))
        axis = 3
    filters = 32
# convolutiona and pooling level 1
    conv_224 = conv_layer(inputs, filters)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)
# convolutiona and pooling level 2
    conv_112 = conv_layer(pool_112, 2*filters)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)
# convolutiona and pooling level 3
    conv_56 = conv_layer(pool_56, 4*filters)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)
# convolutiona and pooling level 4
    conv_28 = conv_layer(pool_28, 8*filters)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)
# convolutiona and pooling level 5
    conv_14 = conv_layer(pool_14, 16*filters)
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)
# Conlovlution and feature concatenation
    conv_7 = conv_layer(pool_7, 32*filters)
# Upsampling with convolution 1
# Upsampling with convolution 1
    up_14 = concatenate([Conv2DTranspose(int(conv_7.shape[3]), (2,2), strides=[2, 2],padding='same')(conv_7),conv_14],axis=axis)
    #up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_14], axis=axis)
    up_conv_14 = conv_layer(up_14, 16*filters)
# Upsampling with convolution 2
    up_28 = concatenate([Conv2DTranspose(int(up_conv_14.shape[3]), (2,2), strides=[2, 2],padding='same')(up_conv_14),conv_28],axis=axis)
#    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), conv_28], axis=axis)
    up_conv_28 = conv_layer(up_28, 8*filters)
# Upsampling with convolution 3
    up_56 = concatenate([Conv2DTranspose(int(up_conv_28.shape[3]), (2,2), strides=[2, 2],padding='same')(up_conv_28),conv_56],axis=axis)
#    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), conv_56], axis=axis)
    up_conv_56 = conv_layer(up_56, 4*filters)
# Upsampling with convolution 4
    up_112 = concatenate([Conv2DTranspose(int(up_conv_56.shape[3]), (2,2), strides=[2, 2],padding='same')(up_conv_56),conv_112],axis=axis)
#    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = conv_layer(up_112, 2*filters)
# Upsampling with convolution 5
    up_224 = concatenate([Conv2DTranspose(int(up_conv_112.shape[3]), (2,2), strides=[2, 2],padding='same')(up_conv_112),conv_224],axis=axis)
    #up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = conv_layer(up_224, filters, dropout_val)
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
Y_Validation = np.load('../data/vali_label'+aug.split('_aug')[-1]+'.npy')#[:,:,:,np.newaxis]#[:700]

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
        ModelCheckpoint('model'+aug+'16ad.h5', monitor='val_loss', save_best_only=True, verbose=0),
    ]


results_03 = model.fit(X_train, Y_train, validation_data=(X_Validation,Y_Validation), batch_size=16, epochs=maxepoch, callbacks=callbacks)

from keras.models import load_model
import pickle
# save the model
model.save('../models/model'+aug+'_U_net.h5')
# save the intermdediate results and training statistics
with open('../models/history'+aug+'_U_net.pickle', 'wb') as file_pi:
    pickle.dump(results_03.history, file_pi, protocol=2)

# Save the predicted labels.
X_test = np.load("../data/prediction_data.npy")
preds_test = model.predict(X_test)
preds_test_t = (preds_test > 0.5).astype(np.uint8)
np.save('../result/preds_test_total'+aug+'_U_net.npy',preds_test_t)
