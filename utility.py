import numpy as np
import cv2

def crop (h, w, scale) :
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)

    return h, w

def getSize(image) :
    if len(image.shape) == 3 :
        h, w, _ = image.shape
    else:
        h, w = image.shape

    return h, w

def getImage (h, w, image) :
    if len(image.shape) == 3 :
        image = image[0:h, 0:w, :]
    else:
        image = image[0:h, 0:w]

    return image


def modcrop (image, scale) :
    h, w = getSize(image)
    h, w = crop(h, w, scale)
    return getImage(h, w, image)

def bicubicInterpolation(image, scale, dim ) :
    donwScale = cv2.resize(image,dsize=(0,0), fx=scale,fy=scale, interpolation = cv2.INTER_CUBIC)
    upScale = cv2.resize(donwScale, dsize = (dim[1], dim[0]), interpolation = cv2.INTER_CUBIC)

    return upScale