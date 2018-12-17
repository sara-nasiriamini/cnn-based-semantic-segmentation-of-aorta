import os
import sys
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.layers import merge
from keras.layers import add
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
import tensorflow as tf

# This method is used to measure Dice Score used 
# as metric to measure the performance of the model
#     - y_pred (model output)
#     - y_true (ground truth)
def dice_coef(y_pred, y_true, epsilon=1e-6): 
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(np.square(y_pred) + np.square(y_true), axes)
    
    return (K.mean(numerator / (denominator + epsilon)))
	
# This method is used to measure Dice Loss based on
# measure Dice Score
#     - y_pred (model output)
#     - y_true (ground truth)
def dice_coef_loss(y_pred, y_true):
	return 1 - dice_coef(y_true, y_pred, epsilon=1e-6)

# This method is used to measure combined loss defined 
# as weighted sum of CCE loss and Dice Loss
#     - y_pred (model output)
#     - y_true (ground truth)
def combined_loss(y_pred, y_true):
	ce_loss = K.categorical_crossentropy(y_pred, y_true)
	dice_loss = dice_coef_loss(y_pred, y_true)
	w1=0.5
	w2=0.5
	loss = w1 * ce_loss + w2 * (dice_loss)
	return loss
	
# Unet Model is used for training and perdiction
#     - pretrained_weights (pretrained weights to load during prediction)
#     - input_size (model input size)
#     - bn (whether to trurn BatchNormalization on (True) or off (False))
def unet2d(pretrained_weights = None, input_size = (256,256,1), bn = True):
	inputs = Input(input_size)
    
	c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
	c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
	if (bn):
		cb1 = BatchNormalization()(c1)
		p1 = MaxPooling2D((2, 2)) (cb1)
	else:
		p1 = MaxPooling2D((2, 2)) (c1)

	c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
	c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
	if (bn):
		cb2 = BatchNormalization()(c2)
		p2 = MaxPooling2D((2, 2)) (cb2)
	else:
		p2 = MaxPooling2D((2, 2)) (c2)

	c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
	c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
	if (bn):
		cb3 = BatchNormalization()(c3)
		p3 = MaxPooling2D((2, 2)) (cb3)
	else:
		p3 = MaxPooling2D((2, 2)) (c3)
	 
	c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
	c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
	if (bn):
		cb4 = BatchNormalization()(c4)
		p4 = MaxPooling2D((2, 2)) (cb4)
	else:
		p4 = MaxPooling2D((2, 2)) (c4)
	#p4 = Dropout(0.2)(p4)
	    
	c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
	c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
	#c5d = Dropout(0.2) (c5d)
	
	u6 = Conv2D(512, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(c5))
	u6 = concatenate([u6, c4])
	c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
	c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
	
	u7 = Conv2D(256, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(c6))
	u7 = concatenate([u7, c3])
	c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
	c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
    
	u8 = Conv2D(128, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(c7))
	u8 = concatenate([u8, c2])
	c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
	c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)

	u9 = Conv2D(64, (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(c8))
	u9 = concatenate([u9, c1])
	c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
	c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)

	outputs = Conv2D(3, (1, 1), activation='softmax') (c9)
	model = Model(input = inputs, output = outputs)

	model.compile(optimizer = Adam(lr = 1e-5), loss = dice_coef_loss, metrics = [dice_coef])
	#model.compile(optimizer = Adam(lr = 1e-4), loss = combined_loss, metrics = [dice_coef])
	#model.compile(optimizer = Adam(lr = 5e-4), loss = 'categorical_crossentropy', metrics = [dice_coef])
	model.summary()
	
	if (pretrained_weights):
	    model.load_weights(pretrained_weights)
	
	return model
