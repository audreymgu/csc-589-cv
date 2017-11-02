# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 09:08:16 2015
A Gaussian pyramid is basically a series of increasingly decimated images, traditionally at downsampling rate r=2. At each level, the image is first blurred by convolving with a Gaussian-like filter to prevent aliasing in the downsampled image. We then move up a level in the Gaussian pyramid by downsampling the image (halving each dimension). 
To build the Laplacian pyramid, we take each level of the Gaussian pyramid and subtract from it the next level interpolated to the same size.

@author: bxiao from http://pauljxtan.com/blog/011315/

Modification and additional comments by ggu
"""
import cv2
import numpy as np
from scipy import misc
from scipy import ndimage
import scipy.signal as sig
import matplotlib.pyplot as plt


# Binomial (5-tap) filter for grayscale images
kernel = (1.0/256)*np.array([[1, 4,  6,  4,  1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6],[4, 16, 24, 16, 4],[1, 4,  6,  4,  1]])

# Processing Functions

def interpolate(image):
    """
    Interpolates an image with an upsampling rate r=2.
    """
    image_up = np.zeros((2*image.shape[0], 2*image.shape[1]))
    # Upsample
    image_up[::2, ::2] = image
    # Blur (we need to scale this up since the kernel has unit area)
    # (The length and width are both doubled, so the area is quadrupled)
    return ndimage.filters.convolve(image_up,4*kernel, mode='constant')

def decimate(image):
    """
    Decimates an image with a downsampling rate r=2.
    """
    # Blur
    image_blur = ndimage.filters.convolve(image,kernel, mode='constant')
    # Downsample
    return image_blur[::2, ::2]                                

def pyramids(image):
    """
    Constructs Gaussian and Laplacian pyramids from a given image.
    
    Arguments:
        image   : the original image
    Return Values:
        G   : Gaussian pyramid
        L   : Laplacian pyramid
    """
    # Initialize pyramids
    G = [image, ]
    L = []

    # Build the Gaussian pyramid to maximum depth
    while image.shape[0] >= 2 and image.shape[1] >= 2:
        image = decimate(image)
        G.append(image)
   
    # Build the Laplacian pyramid
    for i in range(len(G) - 1):
        L.append(G[i] - interpolate(G[i + 1]))

    return G[:-1], L  

def traverse(L,g):
    """
    Reconstructs an image by combining its laplacian pyramid with it gaussian counterpart.
    The gaussian pyramid is reconstructed in the process of traversing the laplacian pyramid, 
    starting with a 2x2 seed from its last layer.
    
    Arguments :
        L   : Laplacian pyramid
        G   : Smallest layer of Gaussian pyramid
    Return Values :
        image : The original image
    """
    output_image = np.empty_like(L[0])
    gauss_previous = g 
    for i in range(-1, -len(L)-1):
        laplace = L[counter-1]
        gauss_expand = interpolate(gauss_previous)
        combined_image = laplace + gauss_expand
        gauss_previous = combined_image
        output_image[0:combined_image.shape[0], 0:combined_image.shape[1]] = combined_image
    return output_image

def blend(A,B,mask):
    """
    Reconstructs an image by combining its laplacian pyramid with it gaussian counterpart.
    The gaussian pyramid is reconstructed in the process of traversing the laplacian pyramid, 
    starting with a 2x2 seed from its last layer.
    
    Arguments :
        L   : Laplacian pyramid
        G   : Smallest layer of Gaussian pyramid
    Return Values :
        image : The original image
    """
    [gpA,lpA] = pyramids(A)
    [gpB,lpB] = pyramids(B)
    [gpM,lpM] = pyramids(mask)
    lpS = []
    for la,lb,gm in zip(lpA, lpB, gpM):
        rows, cols = la.shape
        lS = la * gm + lb * (1 - gm) 
        lpS.append(lS)
    lpS = lpS[::-1]
    output_image = lpS[0]
    for i in range(1,len(lpS)):
        output_image = interpolate(output_image)
        output_image += lpS[i] 
    return output_image
    
# Image Generation

## Import images
apple = misc.imread('../data/apple.jpg',flatten=-1)
orange = misc.imread('../data/orange.jpg',flatten=-1)
mask = misc.imread('../data/mask.jpg',flatten=1)
mask *= (1.0/mask.max())

## Generate Gaussian and Laplacian Pyramids from a specified test image
# [G,L] = pyramids(apple)
# 
blend_img = blend(apple, orange, mask)
# fig, ax = plt.subplots()
# ax.imshow(blend_img,cmap='gray')
# plt.show()

cv2.imwrite('../data/apple_orange.jpg', blend_img)

## Import test images
businessman = misc.imread('../data/businessman.jpg',flatten=-1)
fist = misc.imread('../data/fist_mod.jpg',flatten=-1)
mask = misc.imread('../data/bf_mask.jpg',flatten=1)
mask *= (1.0/mask.max())

## Generate Gaussian and Laplacian Pyramids from a specified test image
# [G,L] = pyramids(apple)
# 
blend_img = blend(businessman, fist, mask)
# fig, ax = plt.subplots()
# ax.imshow(blend_img,cmap='gray')
# plt.show()

cv2.imwrite('../data/businessman_fist2.jpg', blend_img)

## Import test images
thomas = misc.imread('../data/thomas.jpg',flatten=-1)
pacman = misc.imread('../data/pacman.jpg',flatten=-1)
mask = misc.imread('../data/tp_mask2.jpg',flatten=1)
mask *= (1.0/mask.max())

# ## Generate Gaussian and Laplacian Pyramids from a specified test image
# [G,L] = pyramids(apple)
# 
blend_img = blend(thomas, pacman, mask)
# fig, ax = plt.subplots()
# ax.imshow(blend_img,cmap='gray')
# plt.show()

cv2.imwrite('../data/thomas_pacman2.jpg', blend_img)

## Import test images
girl = misc.imread('../data/girl.jpg',flatten=-1)
godzilla = misc.imread('../data/godzilla.jpg',flatten=-1)
mask = misc.imread('../data/gg_mask.jpg',flatten=1)
mask *= (1.0/mask.max())

# ## Generate Gaussian and Laplacian Pyramids from a specified test image
# [G,L] = pyramids(apple)
# 
blend_img = blend(girl, godzilla, mask)
# fig, ax = plt.subplots()
# ax.imshow(blend_img,cmap='gray')
# plt.show()

cv2.imwrite('../data/girl_godzilla.jpg', blend_img)

# ## Generate composite Gaussian display
# rows, cols = apple.shape
# composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
# composite_image[:rows, :cols] = G[0]
# 
# i_row = 0
# for p in G[1:]:
#     n_rows, n_cols = p.shape[:2]
#     composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
#     i_row += n_rows
# 
# ## Output to window
# fig, ax = plt.subplots()
# ax.imshow(composite_image,cmap='gray')
# plt.show()
# 
## Generate composite Laplacian display
# rows, cols = apple.shape
# composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
# composite_image[:rows, :cols] = L[0]
# 
# i_row = 0
# for p in L[1:]:
#     n_rows, n_cols = p.shape[:2]
#     composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
#     i_row += n_rows
# 
# ## Output to window
# fig, ax = plt.subplots()
# ax.imshow(composite_image,cmap='gray')
# plt.show()
#                                           