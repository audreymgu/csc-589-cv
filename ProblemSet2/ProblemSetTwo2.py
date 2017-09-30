import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage

# Problem 2: Histogram Equalization

## Read in the image
image = misc.imread('initial_images/lowcontrast.jpg',flatten=1)
## Create a histogram from 0 to 255 of all original pixel values
histogram, bin_ref = np.histogram(image, 256, density=True)
## Calculate the original cumulative distribution function
cdf = histogram.cumsum()
## Normalize the cumulative distribution function
cdf_n = 0 + ((cdf-cdf.min())*(255-0))/(cdf.max()-cdf.min())
## Use linear interpolation to remap image values
image_e = np.interp(image, bin_ref[:-1], cdf_n)
## Create a histogram from 0 to 255 of all updated pixel values
histogram_e, bin_ref = np.histogram(image_e, 256, density=True)
## Calculate the updated cumulative distribution function
cdf_e = histogram_e.cumsum()
## Save the adjusted image
cv2.imwrite('generated_images/lowcontrast_equalized.jpg', image_e)

## Display the results
plt.subplot(221),plt.imshow(image, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.plot(cdf)
plt.subplot(223),plt.imshow(image_e, cmap = 'gray')
plt.title('Equalized'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.plot(cdf_e)
plt.show()