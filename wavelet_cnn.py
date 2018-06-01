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
from keras.callbacks import ModelCheckpoint
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


def wavelet_mode (shape=(4, 32, 32)) :
	lr = keras.layers.Input(shape=shape)
	
	layer = Conv2D(64, kernel_size=(5,5),strides=(1,1),padding='same',
				 activation='relu',
				 kernel_initializer=initializers.RandomNormal(stddev=np.sqrt(2.0/9)),
				 bias_initializer=keras.initializers.Zeros(),
				 data_format='channels_first')(lr)
	last_layer = layer
	for i in range(10) :
		last_layer = Conv2D(64,
					 kernel_size=(3, 3),
					 strides=(1,1), 
					 activation='relu', 
					 padding='same', 
					 kernel_initializer=initializers.RandomNormal(stddev=np.sqrt(2.0/9/64)),
					 bias_initializer=keras.initializers.Zeros(), data_format='channels_first')(last_layer)

	
	conv_out = Conv2D(4, 
				 kernel_size=(3, 3),
				 strides=(1,1), 
				 padding='same', 
				 kernel_initializer=initializers.RandomNormal(stddev=np.sqrt(2.0/9/64)),
				 bias_initializer=keras.initializers.Zeros(), data_format='channels_first')(last_layer)
	
	output = keras.layers.Add()([conv_out, lr])
	
	model = keras.models.Model(inputs=lr, outputs=output)
	
	wsrcnn_compile(model)

	return model 

def wsrcnn_compile(model) :
	model.compile(optimizer=optimizers.Adam(), loss='mse')

weigth_name = "weights_"
path = 'models/'
def load_model (model_name, path = path, model_compile=wsrcnn_compile) :
	# load the model from data
	
	fm = open(path + model_name)
	model = model_from_json(fm.read())
	fm.close()

	model.load_weights(path + weigth_name + model_name, by_name=False)

	model_compile(model)
	
	return model

def save_model (model, name, path=path) :
	model_json = model.to_json()
	with open('models/' + name, 'w') as f:
		f.write(model_json)

	model.save_weights(path + weigth_name + name)
