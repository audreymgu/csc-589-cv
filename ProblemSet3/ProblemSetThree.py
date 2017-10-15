import glob
import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage
from scipy import signal

# Cross-correlation

## A cross-correlation operation is mechanically identical to convolution, with the important difference that the kernel is not flipped along each axis prior to the beginning of the operation.

def ccor2d(img, kernel, padding_type):
  ### Create a blank output matrix.
  output_img = np.empty(img.shape)
  ### Get dimensions. Rows first, columns second.
  img_dims = img.shape
  kernel_dims = kernel.shape
  ### Determine required vertical and horizontal padding
  v_pad = (kernel_dims[0]-1)/2
  h_pad = (kernel_dims[1]-1)/2
  ### Pad the image
  pad_spec = ((v_pad,v_pad),(h_pad,h_pad))
  pad_img = np.pad(img, pad_width = pad_spec, mode = padding_type)
  ### Correlate image with kernel
  for i in range(0, img_dims[0], 1):
    slice = pad_img[i:kernel_dims[0]+i,:kernel_dims[1]]
    for j in range(0, img_dims[1], 1):
      center = pad_img[i+v_pad,j+h_pad]
      sliding_window = pad_img[i:kernel_dims[0]+i,j:kernel_dims[1]+j]
      output_img[i,j] = operator(sliding_window, kernel)
  return output_img

## A sub-function was made to perform the mathematical operations.

def operator(a,b):
  container = []
  for i in range(0, a.shape[0], 1):
    for j in range(0, a.shape[1], 1):
      container.append(a[i,j]*b[i,j])
  return sum(container) 

# 2D Convolution

## Convolution, as described earlier, is cross-correlation with a flipped kernel. Flipping - as well as color image handling - are contained in this function.

def conv2d(img, kernel, padding, color):
  flip_k = np.flip(kernel,0)
  flip_k = np.flip(flip_k,1)
  ### Split color image into channels before processing
  if color == 1:
    img[:,:,0] = ccor2d(img[:,:,0], flip_k, padding)
    img[:,:,1] = ccor2d(img[:,:,1], flip_k, padding)
    img[:,:,2] = ccor2d(img[:,:,2], flip_k, padding)
    return img
  ### If grayscale, process as normal
  else:
    return ccor2d(img, flip_k, padding)

# Gaussian Filtering
  
def gauss(sigma, size, size_y=0):
  ### Define a linear set of values, with number of values determined by the input parameter 'size'
  x_ax = np.linspace(-1, 1, size)
  ### Account for non-regular y-dimensions
  if size_y:
    y_ax = np.linspace(-1, 1, size_y)
  else:
    y_ax = np.linspace(-1, 1, size)
  ### Create a square meshgrid using above value sets
  x, y = np.meshgrid(x_ax, y_ax)
  ### Create gaussian approximation with requested deviation
  kernel = np.exp(-(x**2/(2*sigma**2)+y**2/(2*sigma**2)))
  return kernel / np.sum(kernel)
  
# Low-pass Filter

## A filter that attenuates high frequency information.

def lo_pass(img, fname, sigma, color, size=15, padding='symmetric', saving=True):
  gaussian_filter = gauss(sigma, size)
  processed_img = conv2d(img, gaussian_filter, padding, color)
  if saving:
    cv2.imwrite('generated_images/'+fname+'_lo'+str(sigma)+'.bmp', processed_img)
  else:
    pass
  return processed_img
  
# High-pass Filter

## A filter that attenuates low frequency information.

def hi_pass(img, fname, sigma, color, size=15, padding='symmetric', saving=True, complement=True):
  original = np.copy(img)
  copy = np.copy(img)
  lo_img = lo_pass(copy, fname, sigma, color, size, padding, complement)
  if complement:
    print 'Low-pass filter of '+str(fname)+' saved. Now subtracting from original image to obtain high-pass result.'
  else:
    pass
  processed_img = np.subtract(original, lo_img)
  if saving:
    cv2.imwrite('generated_images/'+fname+'_hi'+str(sigma)+'.bmp', processed_img)
    print 'High-pass result of '+str(fname)+' saved.'
  else:
    pass
  return processed_img

# Applying Filters

## A concise iterative approach to filtering images using blob was tested, but ultimately dropped when hi_pass was not proprely saving results when called within this loop. The code is retained below, should the issue be resolved in the future.
# for filename in glob.glob('initial_images/*.bmp'):
#   selected = cv2.imread(filename, -1)
#   hi_img = hi_pass(selected, fname=filename, sigma=5, color=1, size=7, padding='symmetric', saving=True, complement=True)
#   cv2.imwrite('generated_images/'+filename+'hi'+str(5)+'.bmp', hi_img)

## There was some degree of fiddling of various filtering settings to produce high-pass and low-pass images that produced reasonable results when combined. The last settings used are retained here.

### Take the high-pass and low-pass of bicycle.bmp
img = cv2.imread('initial_images/bicycle.bmp', -1)
lo_pass(img, fname='bicycle', sigma=10, color=1, size=15, padding='symmetric', saving=True)
hi_pass(img, fname='bicycle', sigma=5, color=1, size=7, padding='symmetric', saving=True, complement=False)

### Take the high-pass and low-pass of bird.bmp
img = cv2.imread('initial_images/bird.bmp', -1)
lo_pass(img, fname='bird', sigma=10, color=1, size=15, padding='symmetric', saving=True)
hi_pass(img, fname='bird', sigma=3, color=1, size=7, padding='symmetric', saving=True, complement=False)

### Take the high-pass and low-pass of cat.bmp
img = cv2.imread('initial_images/cat.bmp', -1)
lo_pass(img, fname='cat', sigma=10, color=1, size=15, padding='symmetric', saving=True)
hi_pass(img, fname='cat', sigma=3, color=1, size=7, padding='symmetric', saving=True, complement=False)

### Take the high-pass and low-pass of dog.bmp
img = cv2.imread('initial_images/dog.bmp', -1)
lo_pass(img, fname='dog', sigma=10, color=1, size=15, padding='symmetric', saving=True)
hi_pass(img, fname='dog', sigma=3, color=1, size=7, padding='symmetric', saving=True, complement=False)

### Take the high-pass and low-pass of einstein.bmp
img = cv2.imread('initial_images/einstein.bmp', -1)
lo_pass(img, fname='einstein', sigma=10, color=1, size=15, padding='symmetric', saving=True)
hi_pass(img, fname='einstein', sigma=3, color=1, size=7, padding='symmetric', saving=True, complement=False)

### Take the high-pass and low-pass of fish.bmp
img = cv2.imread('initial_images/fish.bmp', -1)
lo_pass(img, fname='fish', sigma=10, color=1, size=15, padding='symmetric', saving=True)
hi_pass(img, fname='fish', sigma=3, color=1, size=7, padding='symmetric', saving=True, complement=False)

### Take the high-pass and low-pass of marilyn.bmp
img = cv2.imread('initial_images/marilyn.bmp', -1)
lo_pass(img, fname='marilyn', sigma=10, color=1, size=15, padding='symmetric', saving=True)
hi_pass(img, fname='marilyn', sigma=3, color=1, size=7, padding='symmetric', saving=True, complement=False)

### Take the high-pass and low-pass of motorcycle.bmp
img = cv2.imread('initial_images/motorcycle.bmp', -1)
lo_pass(img, fname='motorcycle', sigma=10, color=1, size=15, padding='symmetric', saving=True)
hi_pass(img, fname='motorcycle', sigma=3, color=1, size=7, padding='symmetric', saving=True, complement=False)

### Take the high-pass and low-pass of plane.bmp
img = cv2.imread('initial_images/plane.bmp', -1)
lo_pass(img, fname='plane', sigma=10, color=1, size=15, padding='symmetric', saving=True)
hi_pass(img, fname='plane', sigma=3, color=1, size=7, padding='symmetric', saving=True, complement=False)

### Take the high-pass and low-pass of submarine.bmp
img = cv2.imread('initial_images/submarine.bmp', -1)
lo_pass(img, fname='submarine', sigma=10, color=1, size=15, padding='symmetric', saving=True)
hi_pass(img, fname='submarine', sigma=3, color=1, size=7, padding='symmetric', saving=True, complement=False)

# Image Hybridization

### Combine dog (high-pass) with cat (low-pass). The high-pass information was multiplied by three to increase its prominence.
hi_img = cv2.imread('generated_images/dog_hi5.bmp', -1)
lo_img = cv2.imread('generated_images/cat_lo10.bmp', -1)
composite = hi_img*3 + lo_img
cv2.imwrite('generated_images/dog_cat_d5_boost.bmp', composite)

### Combine einstein (high-pass) with marilyn (low-pass).
hi_img = cv2.imread('generated_images/einstein_hi5.bmp', -1)
lo_img = cv2.imread('generated_images/marilyn_lo10.bmp', -1)
composite = hi_img + lo_img
cv2.imwrite('generated_images/einstein_marilyn_d5.bmp', composite)

### Combine plane (high-pass) with bird (low-pass).
hi_img = cv2.imread('generated_images/plane_hi5.bmp', -1)
lo_img = cv2.imread('generated_images/bird_lo10.bmp', -1)
composite = hi_img + lo_img
cv2.imwrite('generated_images/plane_bird_d5.bmp', composite)

### Combine fish (high-pass) with submarine (low-pass).
hi_img = cv2.imread('generated_images/fish_hi5.bmp', -1)
lo_img = cv2.imread('generated_images/submarine_lo10.bmp', -1)
composite = hi_img + lo_img
cv2.imwrite('generated_images/fish_submarine_d5.bmp', composite)

## Combining the low frequency information from the motorcycle with the high frequency information from the bicycle resulted in an image that had significant artifacting. This version was dropped in favor of the converse.
# hi_img = cv2.imread('generated_images/motorcycle_hi5.bmp', -1)
# lo_img = cv2.imread('generated_images/bicycle_lo10.bmp', -1)
# composite = hi_img + lo_img
# cv2.imwrite('generated_images/motorcycle_bicycle_d5.bmp', composite)

### Combine bicycle (high-pass) with motorcycle (low-pass)
hi_img = cv2.imread('generated_images/bicycle_hi5.bmp', -1)
lo_img = cv2.imread('generated_images/motorcycle_lo10.bmp', -1)
composite = hi_img + lo_img
cv2.imwrite('generated_images/bicycle_motorcycle_d5.bmp', composite)



