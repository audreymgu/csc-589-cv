import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage

# Problem 1: Gaussian Blurring and Discrete Fourier Transformations

## Image 1: Zebra

### Read in the image
zebra = misc.imread('initial_images/zebra.jpg',flatten=1)
### Apply ndimage's built-in gaussian filter
z_blur = ndimage.gaussian_filter(zebra,10)
### Save Result
cv2.imwrite('generated_images/zebra_blur.jpg', z_blur)

### Use numpy's 2D fast fourier transform
z_fft = np.fft.fft2(zebra)
### Shift zero frequency to the center of the Fourier spectrum
z_fft = np.fft.fftshift(z_fft)
### Calculate the magnitude
z_magnitude = np.log(np.abs(z_fft))
### Save Result, multiplying spectrum image by 20 for visibility
cv2.imwrite('generated_images/zebra_fft.jpg', z_magnitude*20)

## Image 2: Cheetah

### Read in the image
cheetah = misc.imread('initial_images/cheetah.jpg',flatten=1)
### Apply ndimage's built-in gaussian filter
c_blur = ndimage.gaussian_filter(cheetah,10)
### Save Result
cv2.imwrite('generated_images/cheetah_blur.jpg', c_blur)

### Use numpy's 2D fast fourier transform
c_fft = np.fft.fft2(cheetah)
### Shift zero frequency to the center of the Fourier spectrum
c_fft = np.fft.fftshift(c_fft)
### Calculate the magnitude
c_magnitude = np.log(np.abs(c_fft))
### Save Result, multiplying spectrum image by 20 for visibility
cv2.imwrite('generated_images/cheetah_fft.jpg', c_magnitude*20)

## Display results as titled images in 2 row and 3 column grid
plt.subplot(231),plt.imshow(zebra, cmap = 'gray')
plt.title('Zebra (Original)'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(z_blur, cmap = 'gray')
plt.title('Zebra (Blur)'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(z_magnitude, cmap = 'gray')
plt.title('Zebra (DFT)'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(cheetah, cmap = 'gray')
plt.title('Cheetah (Original)'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(c_blur, cmap = 'gray')
plt.title('Cheetah (Blur)'), plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(c_magnitude, cmap = 'gray')
plt.title('Cheetah (DFT)'), plt.xticks([]), plt.yticks([])
plt.show()