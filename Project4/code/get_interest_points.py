# Import packages
import math
import numpy as np
from scipy import misc
from scipy import ndimage
import cv2

# Functions

def get_interest_points(img):
	### Create initial seed values for blurring
	sigma = 1.6
	k = math.sqrt(2)
	### Create initial images for each octave
	img_0 = img
	img_1 = cv2.pyrDown(img_0)
	img_2 = cv2.pyrDown(img_1)
	img_3 = cv2.pyrDown(img_2)
	### Create empty lists to hold each of the four gaussian octaves
	octave_0 = []
	octave_1 = []
	octave_2 = []
	octave_3 = []
	### Create empty lists to hold each of the four DoG octaves
	dog_octave_0 = []
	dog_octave_1 = []
	dog_octave_2 = []
	dog_octave_3 = []
	### Generate five scales per octave with sigma, increasingly multiplicatively by k
	for i in range(4):
		for j in range(5):
			blur_img = ndimage.filters.gaussian_filter(eval(str('img_')+str(i)), sigma*k**(j+i*2))
			eval(str('octave_')+str(i)).append(blur_img)
	### Find the Difference of Gaussians for each octave
	for i in range(4):
		for j in range(4):
			dog_img = eval(str('octave_')+str(i)+str('[')+str(j)+str(']')) - eval(str('octave_')+str(i)+str('[')+str(j+1)+str(']'))
			eval(str('dog_octave_')+str(i)).append(dog_img)
	### Find local extrema over scale and space
# Testbench

## Basic Testing
sigma = 1.6
### Import test image
image1 = misc.imread('/Users/gdes/Downloads/Project4Feature/code/vis.jpg',flatten=1)
### Manipulate test image
blur_img = ndimage.filters.gaussian_filter(image1, sigma*1.6**(1+2*2))
### Save test image
cv2.imwrite('/Users/gdes/Downloads/Project4Feature/code/test.jpg', blur_img)

## Function Testing
get_interest_points(image1)
