import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage as nd

# Problem 4: Edge Detection

## Create 1D Sobel filters
sobel_d = np.array([-1,0,1])
sobel_s = np.array([1,2,1])

## Read in the image
rect = misc.imread('initial_images/rectangle.jpg',flatten=1)

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

## Use imfilter to process images

### Sobel Operator: X Direction
sobelimg_x = imfilter(rect, sobel_d, sobel_s, 'zero')

### Sobel Operator: Y Direction
sobelimg_y = imfilter(rect, sobel_s, sobel_d, 'zero')

## Composite both X-direction and Y-direction sobel passes
sobelimg = np.hypot(sobelimg_x, sobelimg_y)

## Save Result
cv2.imwrite('generated_images/edges.jpg', sobelimg)







