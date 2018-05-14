import keras
from keras import backend as K
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.layers import Conv2D
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
def get_input_images (img, scale = 3) :
	original = img  #utility.modcrop(img, scale)
	height, width = utility.getSize(original)
	bicubic = utility.bicubicInterpolation(original, 1./scale, (height,width))
	
	return original, bicubic

def get_padding ( sample_size = 33, label_size = 21) :
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
def generate_patches(image, patch_size = 33, label_size = 21, scale = 3, stride = 14) :
	#Generate low resolution image
	label, sample = get_input_images(image, scale)
	height, width = utility.getSize(label)
	
	padding = get_padding(patch_size, label_size)
	
	samples = []
	labels = []
	#Calculate subimages
	for h in range(0, height - patch_size, stride ) :
		for w in range(0, width - patch_size, stride) :
			sub_sample = sample[h : h + patch_size, w : w + patch_size]
			sub_label = label[h + padding : h + padding + label_size, w + padding : w +  padding + label_size]
			
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
def image_patches(images, sample_size = 33, label_size = 21, scale = 3, stride = 14) :
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
	x = np.asarray(sample, dtype=np.float32)
	y = np.asarray(label, dtype=np.float32)

	train = x / 255
	test = y / 255
	
	return train, test

def reshape(train, test, train_size=33, test_size=21, ch=3) :
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
def patch_to_image(patches, height, width, padding=0, sample_size=33, label_size=21, stride=14, ch=3) :
	count = 0
	zeros = np.zeros((height, width,ch))
		
	for h in range(0, height - sample_size, stride ) :
		for w in range(0, width - sample_size, stride) :
			zeros[h + padding : h + padding + label_size, w + padding : w +  padding + label_size] = patches[count]
			count = count + 1
			
	assert(count == len(patches))
	return zeros

'''
 PSNR of images
'''
def psnr (y_pred, y) :
	t = K.mean(K.square(y_pred - y))
	return -10. * K.log(t)


kernel_ini = initializers.RandomNormal(mean=0.0, stddev=1e-3, seed=None)
bias_ini = keras.initializers.Zeros()

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 

def srcnn_mode(net=[64,32,3], flt=[9,1,5], kernel_ini=kernel_ini, bias_ini=bias_ini) :
	model = Sequential()
	conv1 = model.add(Conv2D(net[0], kernel_size=(flt[0], flt[0]), strides=(1, 1),data_format="channels_last",
				 activation='relu',
				 input_shape=(33, 33, 3),
				 kernel_initializer=kernel_ini,
				 bias_initializer=bias_ini))
	
	model.add(Conv2D(net[1], kernel_size=(flt[1], flt[1]), strides=(1, 1),data_format="channels_last",
				 activation='relu',
				 kernel_initializer=kernel_ini,
				 bias_initializer=bias_ini))

	model.add(Conv2D(net[2], kernel_size=(flt[2], flt[2]), strides=(1, 1),data_format="channels_last",
				 kernel_initializer=kernel_ini,
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
def predict_image (model, img, strides=21, sample_size = 33, label_size = 21, scale = 3, stride = 21, channels=3, batch_size=128) :
	subBic, subOrg = generate_patches(img, 
									  patch_size = sample_size, 
									  label_size = label_size, 
									  scale = scale, 
									  stride = stride )

	subOrg, subBic = normalize(subOrg, subBic)    
	pred = subBic.reshape(-1, 33, 33, channels)
	
	im = model.predict(pred, batch_size=batch_size)

	pad = get_padding(sample_size, label_size)
	
	h, w = utility.getSize(img)
	image = patch_to_image(im, h, w, pad, sample_size, label_size, stride=stride)
	return np.clip(image, 0, 1)
 
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
