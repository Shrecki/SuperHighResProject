# %load srcnn.py
import random

import tensorflow as tf
from tensorflow.python.client import device_lib

import keras
from keras import backend as K
CUDA_VISIBLE_DEVICES = 1

from keras import regularizers

from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.layers import Conv2D
from keras.layers import ZeroPadding2D
from keras.layers import BatchNormalization
from keras import losses
from keras import initializers
from keras import optimizers
from keras.models import model_from_json

import numpy as np
import math
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter
from os import listdir
import random

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.Session(config=config)
keras.backend.set_session(session)

K.tensorflow_backend.set_session(tf.Session(config=config))

import utility

from srPreprocessing import generate_patches
from srPreprocessing import patch_to_image

def psnr (y_pred, y) :
	t = K.mean(K.square(y_pred - y))
	return -10. * K.log(t)


kernel_ini = initializers.RandomNormal(mean=0.0, stddev=1e-4, seed=None)
bias_ini = keras.initializers.Zeros()

adam = optimizers.Adam(lr=0.00001)

def srcnn_mode(net=[64,32,3], flt=[9,1,5], kernel_ini=kernel_ini, bias_ini=bias_ini) :
	model = Sequential()
	model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),data_format="channels_last",
				 activation='relu',
				 input_shape=(32, 32, 1),
				 padding='same',
				 bias_initializer=bias_ini))
	for i in range(10):
		model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),data_format="channels_last",
				 activation='relu',
				 padding='same',
				 kernel_initializer=initializers.RandomNormal(stddev=np.sqrt(2.0/9/64)),
				 bias_initializer=bias_ini))
	model.add(Conv2D(1, kernel_size=(3, 3), strides=(1, 1),data_format="channels_last",
			 padding='same',
			 kernel_initializer=initializers.RandomNormal(stddev=np.sqrt(2.0/9/64)),
			 bias_initializer=bias_ini))

	
	print(model.summary())
	return model

def srcnn_compile (model, loss="mean_squared_error", metrics=[psnr], opt = adam ) : 
	model.compile(loss=loss, metrics=metrics, optimizer=adam) 

'''
 Predict image given a model
	param   model : Model
   param image : Original image to predit
   param sample_size : Patches size
   param label_size : Center patch size, validation set size
   param scale : Interpolation scale
   param stride : Stride
   param channels
'''
def predict_image (model, image, patch_size = 32, scale = 4., stride = 21, batch_size=64):
	subBic, subOrg = generate_patches(image, 
								  patch_size = patch_size, 
								  scale = scale, 
								  stride = stride )
	
	subOrg, subBic = normalize(subOrg, subBic)
	subOrg, subBic = reshape(subOrg, subBic)

	pred = model.predict(subBic, batch_size)
	pred = pred.clip(0,1)

	h, w = utility.getSize(image)
	image = patch_to_image(pred, h, w, patch_size, stride=stride)
	
	image = image*255
	image = image.astype('uint8')
	
	return image
 
weigth_name = "weights_"
path = 'models/'
def load_model (model_name, path = path, model_compiler=srcnn_compile) :
	# load the model from data
	
	fm = open(path + model_name)
	model = model_from_json(fm.read())
	fm.close()

	model.load_weights(path + weigth_name + model_name, by_name=False)

	model_compiler(model)
	
	return model

def save_model (model, name, path=path) :
	model_json = model.to_json()
	with open('models/' + name, 'w') as f:
		f.write(model_json)

	model.save_weights(path + weigth_name + name)

'''
 Normalize train and test set
'''
def normalize(sample, label) : 
	x = np.asarray(sample)
	y = np.asarray(label)
	
	train = x / 255.
	test = y / 255.
	
	return train, test
'''
 Reshape train and test set
'''
def reshape(train, test, train_size=32, test_size=32, ch=1) :
	train = train.reshape(-1, train_size, train_size, ch)
	test = test.reshape(-1, test_size, test_size, ch)
	
	return train, test