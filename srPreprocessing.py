from PIL import Image, ImageFilter
from os import listdir

import numpy as np
import math
import matplotlib.pyplot as plt

import pywt

import utility


def load_images (img_folder, nb_images=0) :
	obj_files = listdir(img_folder )
	imgs = []

	if (nb_images == 0) :
		nb_images = len(obj_files)

	for i in range(0, nb_images):
		n = obj_files[i]
		img = Image.open(img_folder + n)
		img = img.convert('YCbCr')
		imgs.append(np.asarray(img)[:,:,0])

	return imgs
'''
 Return the original image and a bicubic interpolation cropped to the same size
'''
def get_input_images (original, scale = 4.) :
	#original = utility.modcrop(img, scale)
	height, width = utility.getSize(original)
	bicubic = utility.bicubicInterpolation(original, 1/scale, (height,width))
	
	return original, bicubic

''' Generate patches of a given image

   param image : Image to extract patches
   param patch_size : Patch size
   param scale : Interpolation scale
   param stride : Stride
'''
def generate_patches(image, patch_size = 32, scale = 4., stride = 14) :
	#Generate low resolution image
	label, sample = get_input_images(image, scale)
	height, width = utility.getSize(label)
	
	samples = []
	labels = []

	#Calculate subimages
	for h in range(0, height - patch_size, stride ) :
		for w in range(0, width - patch_size, stride) :
			sub_sample = sample[h : h + patch_size, w : w + patch_size]
			sub_label = label[h : h + patch_size, w : w + patch_size]

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
def image_patches(images, patch_size = 32, scale = 4., stride = 14) :
	samples = []
	labels = []
	
	for img in images :
		smp, lbs = generate_patches(img, patch_size, scale, stride)
		samples += smp
		labels += lbs
		
	return samples, labels

'''
 Plot image to compare
'''
def plot_images (images, titles, size= (10,5), ch=1) :
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

def plot4subbands(im1, im2, im3, im4, ch=1) :
	size = (10, 10)
	fig = plt.figure(figsize=size)
	
	plt.subplot(221)
	plt.title("Approximation")
	plt.imshow(im1, cmap=plt.get_cmap('gray'))
	plt.subplot(222)
	plt.title("Horizontal")
	plt.imshow(im2, cmap=plt.get_cmap('gray'))
	plt.subplot(223)
	plt.title("Vertical")
	plt.imshow(im3, cmap=plt.get_cmap('gray'))
	
	plt.subplot(224)
	plt.title("Diagonal")
	plt.imshow(im4, cmap=plt.get_cmap('gray'))

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

def plot_waveletTrans(wt, ch=1) :
	plot4subands(wt[0], wt[1][0], wt[1][1], wt[1][2], ch)

''' 
	Generate patches of a given image

   param image : Image to extract patches
   param patch_size : Patch size
   param label_size : Center patch size, validation set size
   param scale : Interpolation scale
   param stride : Stride
'''
def patch_to_image(patches, height, width, label_size=32, stride=14, ch=1) :
	count = 0
	zeros = np.zeros((height, width))
	
	patches = [ptch[:,:,0] if len(ptch.shape) > 2 else ptch for ptch in patches]

	for h in range(0, height - label_size, stride ) :
		for w in range(0, width - label_size, stride) :
			zeros[h : h  + label_size, w : w + label_size] = patches[count]
			count = count + 1
			
	assert(count == len(patches))
	return zeros

def appendSubbands(l1,l2,l3,l4, dwt):
	l1 = np.append(l1 , dwt[0])
	l2 = np.append(l2, dwt[1][0])
	l3 = np.append(l3, dwt[1][1])
	l4 = np.append(l4, dwt[1][2])
	return (l1,l2,l3,l4)

def get_wavelets_input(labels, samples) :
    
    label_set = [] 
    train_set = []
    for i in range(len(labels)) :
        x = samples[i]
        y = labels[i]
    
    
        dwt_hd = get_wavelets(y)
        
        dwt_lw = get_wavelets(x)
        
        label_set.append(dwt_hd)
        train_set.append(dwt_lw)
    
    index = np.random.permutation(len(label_set))
    
    label_set = np.asarray(label_set)
    train_set = np.asarray(train_set)
    
    return label_set[index], train_set[index]

def get_wavelets(img) :
    
    dwt = pywt.dwt2(img, 'haar')
    
    dwt = np.asfarray([dwt[0], dwt[1][0], dwt[1][1], dwt[1][2]])
        
    return dwt

def iwavelet(patch, max_value=255) :
    dwt = (patch[0], (patch[1], patch[2], patch[3]))
    inverse = pywt.idwt2(dwt, 'haar')
    return np.clip(inverse, 0, inverse.max())