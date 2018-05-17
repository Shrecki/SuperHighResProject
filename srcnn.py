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

def to_ycbcr(img) :
	return img.convert('L', (0.2989, 0.5870, 0.1140, 0))

def load_images (img_folder, channels=3) :
	if (type(img_folder) != list) :
		raise ValueError("No image folder list")
	
	img_format = ["bmp", "jpg", "png", "JPEG"]
	
	#List image files for each folder
	img_files = [folder + "/" + img for folder in img_folder for img in listdir(folder) if img.split(".")[-1] in img_format]

	if(len(img_files) == 0) :
		raise ValueError("No image file")

	#Load images
	pil_img = []
	for img_file in img_files:
		img = Image.open(img_file)
		if (img.mode == "RGB") :
			copy = img.copy()

		pil_img.append(copy)
		img.close()

	#pil_img = [Image.open(img_file) for img_file in img_files ]

	if (channels == 1) :
		pil_img = [to_ycbcr(img) for img in pil_img]
		
	imgs = [np.asarray(image) for image in pil_img]
	
	return imgs

'''
 Return the original image and a bicubic interpolation cropped to the same size
'''
def get_input_images (img, scale = 4) :
	original = utility.modcrop(img, scale)
	height, width = utility.getSize(original)
	bicubic = utility.bicubicInterpolation(original, 1./scale, (height,width))
	
	return original, bicubic

def get_padding ( sample_size = 32, label_size = 32) :
	return int(abs(sample_size - label_size)/2)

''' 
Image center following the padding
'''
def center (img, size) :
	img_size = utility.getSize(img)
	
	if(img_size[0] != img_size[1]) :
		raise ValueError("No squared images")
	
	if(img_size[0] <= size) :
		return img
	
	pad = get_padding(img_size[0], size)
	
	return img[pad : pad + size, pad: pad + size]

''' Generate patches of a given image

   param image : Image to extract patches
   param patch_size : Patch size
   param label_size : Center patch size, validation set size
   param scale : Interpolation scale
   param stride : Stride
'''
def generate_patches(image, patch_size = 32, label_size = 32, scale = 3, stride = 14) :
	#Generate low resolution image
	label, sample = get_input_images(image, scale)
	height, width = utility.getSize(label)
	
	samples = []
	labels = []
	#Calculate subimages
	for h in range(0, height - patch_size, stride ) :
		for w in range(0, width - patch_size, stride) :
			sub_sample = sample[h : h + patch_size, w : w + patch_size]

			sub_label = label[h : h + label_size, w : w + label_size]           
			samples.append(sub_sample)
			labels.append(sub_label)
			
	return samples, labels

''' 
Generate patches of a list of image

	param image : Image to extract patches
	param patch_size : Patch size
	param label_size : Center patch size, validation set size
	param scale : Interpolation scale
	param stride : Stride
'''
def image_patches(images, sample_size = 32, label_size = 32, scale = 3, stride = 14) :
	samples = []
	labels = []
	
	for img in images :
		smp, lbs = generate_patches(img, sample_size, label_size, scale, stride)
		samples += smp
		labels += lbs
		
	return samples, labels

'''
 Normalize train and test set
'''
def normalize(sample, label) : 
	x = np.asarray(sample)
	y = np.asarray(label)
    
	train = x / 255.
	test = y / 255.
	
	return train, test

def reshape(train, test, train_size=32, test_size=32, ch=3) :
	train = train.reshape(-1, train_size, train_size, ch)
	test = test.reshape(-1, test_size, test_size, ch)
	
	return train, test

'''
 Plot image to compare
 '''
def plot_images (images, titles, size= (10,5), ch=3) :
	nb_img = len(images) 
	assert (nb_img == len(titles))
	
	subplot = "1" + str(len(images))
	
	fig = plt.figure(figsize=size)
	
	for i in range(nb_img) :
		plt.subplot(subplot + str(i))
		if ( ch == 1) :
			plt.imshow(images[i], cmap=plt.get_cmap('gray'))
		else :
			plt.imshow(images[i])
		plt.title(titles[i])

	plt.show()

''' 
	Generate patches of a given image

   param image : Image to extract patches
   param patch_size : Patch size
   param label_size : Center patch size, validation set size
   param scale : Interpolation scale
   param stride : Stride
'''
def patch_to_image(patches, height, width, padding=0, sample_size=32, label_size=32, stride=14, ch=3) :
	count = 0
	zeros = np.zeros((height, width,ch))
		
	for h in range(0, height - sample_size, stride ) :
		for w in range(0, width - sample_size, stride) :
			zeros[h : h  + label_size, w : w + label_size] = patches[count]
			count = count + 1
			
	assert(count == len(patches))
	return zeros

'''
 PSNR of images
'''
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
				 input_shape=(32, 32, 3),
				 padding='same',
				 kernel_regularizer = regularizers.l2(0.0001),
				 #kernel_initializer=initializers.RandomNormal(stddev=np.sqrt(2.0/9)),
				 bias_initializer=bias_ini))
	for i in range(10):
		model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),data_format="channels_last",
				 activation='relu',
				 padding='same',
				 #kernel_regularizer=regularizers.l2(0.0001),
				 kernel_initializer=initializers.RandomNormal(stddev=np.sqrt(2.0/9/64)),
				 bias_initializer=bias_ini))
	model.add(Conv2D(3, kernel_size=(3, 3), strides=(1, 1),data_format="channels_last",
			 padding='same',
			 #kernel_regularizer=regularizers.l2(0.0001),
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
def predict_image (model, image, sample_size = 32, label_size = 32, scale = 4, stride = 21, channels=3, batch_size=64):
    org, bic = get_input_images(image, 4)
    subBic, subOrg = generate_patches(image, 
                                  patch_size = 32, 
                                  label_size = 32, 
                                  scale = 4, 
                                  stride = stride )
    
    subOrg, subBic = normalize(subOrg, subBic)

    pred = model.predict(subBic, batch_size)
    pred = pred.clip(0,1)

    h, w = utility.getSize(image)
    image = patch_to_image(pred, h, w, 0, sample_size, label_size, stride=stride)
    
    image = image*255
    image = image.astype('uint8')
    
    return image
 
weigth_name = "weights_"
path = 'models/'
def load_model (model_name, path = path) :
	# load the model from data
	
	fm = open(path + model_name)
	model = model_from_json(fm.read())
	fm.close()

	model.load_weights(path + weigth_name + model_name, by_name=False)

	srcnn_compile(model)
	
	return model

def save_model (model, name, path=path) :
	model_json = model.to_json()
	with open('models/' + name, 'w') as f:
		f.write(model_json)

	model.save_weights(path + weigth_name + name)