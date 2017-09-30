import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage as nd

# Problem 3: Separable Filters

## Create 1D Filters
gauss = np.array([.0625,.25,.375,.25,.0625])
box = np.array([0.2,0.2,0.2,0.2,0.2])
sobel_d = np.array([-1,0,1])
sobel_s = np.array([1,2,1])

## Read in the image
cheetah = misc.imread('initial_images/cheetah.jpg',flatten=1)

## Define image filter
def imfilter(img, h, v, mode):
  first = component_process(img,h,mode,1)
  second = component_process(first,v,mode,0)
  return second

## Define filter sub-component
def component_process(img, h, mode, axisval):
  if mode == 'zero':
    valset = 0.0
    modeset = 'constant'
  elif mode == 'constant':
    valset = 1.0
    modeset = 'constant'
  elif mode == 'clamp':
    modeset = 'nearest'
    valset = 0.0
  elif mode == 'wrap':
    modeset = 'wrap'
    valset = 0.0
  else:
    valset = 0.0
    modeset = 'constant'
  process = nd.filters.convolve1d(img, h, axis=axisval, mode=modeset, cval=valset, origin=0)
  return process

## Use imfilter to process images and imwrite to save them

### Gaussian Filter
gaussimg = imfilter(cheetah, gauss, gauss,'zero')
gaussimg_h = component_process(cheetah, gauss, 'zero', 1)
gaussimg_v = component_process(cheetah, gauss, 'zero', 0)
cv2.imwrite('generated_images/gaussian.jpg', gaussimg)
cv2.imwrite('generated_images/gaussian_horizontal.jpg', gaussimg_h)
cv2.imwrite('generated_images/gaussian_vertical.jpg', gaussimg_v)

### Box Filter
boximg = imfilter(cheetah, box, box, 'constant')
boximg_h = component_process(cheetah, box, 'constant', 1)
boximg_v = component_process(cheetah, box, 'constant', 0)
cv2.imwrite('generated_images/box.jpg', boximg)
cv2.imwrite('generated_images/box_horizontal.jpg', boximg_h)
cv2.imwrite('generated_images/box_vertical.jpg', boximg_v)

### Sobel Operator: X Direction
sobelimg_x = imfilter(cheetah, sobel_d, sobel_s, 'clamp')
sobelimg_xd = component_process(cheetah, sobel_d, 'clamp', 1)
sobelimg_xs = component_process(cheetah, sobel_s, 'clamp', 0)
cv2.imwrite('generated_images/sobel_x.jpg', sobelimg_x)
cv2.imwrite('generated_images/sobel_xderivative.jpg', sobelimg_xd)
cv2.imwrite('generated_images/sobel_xsmoothing.jpg', sobelimg_xs)

### Sobel Operator: Y Direction
sobelimg_y = imfilter(cheetah, sobel_s, sobel_d, 'wrap')
sobelimg_yd = component_process(cheetah, sobel_d, 'wrap', 1)
sobelimg_ys = component_process(cheetah, sobel_s, 'wrap', 0)
cv2.imwrite('generated_images/sobel_y.jpg', sobelimg_y)
cv2.imwrite('generated_images/sobel_yderivative.jpg', sobelimg_yd)
cv2.imwrite('generated_images/sobel_ysmoothing.jpg', sobelimg_ys)




  

  