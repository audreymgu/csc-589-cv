import cv2
import numpy as np
from matplotlib import pyplot as plt

# Section 1
# Setup. Already completed.

# Section 2
# Define Matrix M and Vectors a, b, c.
# Matrices
M = np.array([(1,2,3),(4,5,6),(7,8,9),(0,2,2)])

# Vectors
a = np.array([1,1,0])
b = np.array([-1,2,5])
c = np.array([0,2,3,2])

# Find the dot product (also known as the inner product) of vectors a and b. Save the resulting value as variable 'aDotb'.
aDotb = np.inner(a,b)
print aDotb

# Find the element-wise product of a and b.
a.shape = (3,1) # Convert to column
b.shape = (3,1) # Convert to column
aElementb = np.multiply(a,b)
print aElementb

# Find (a^Tb)^Ma.
aMatrixb = np.outer(b,a)
print aMatrixb

# Without using a loop, multiply each row and column of M element-wise by a.
a.shape = (1,3) # Convert to row
M = np.multiply(M,a)
print M

# Without using a loop, sort all of the values of the new M from (e) in increasing order and plot them in your report.
M = np.sort(M, axis=None)
M.shape = (4,3)
print M

# Section 3
# Read in the images 'image1.jpg' and 'image2.jpg'.
img1 = cv2.imread('image1.jpg',-1)
img2 = cv2.imread('image2.jpg',-1)

# Convert the images to double precision and rescale them to stretch from a minimum value of 0 to maximum value of 1.
img1 = np.float32(img1)
img2 = np.float32(img2)
print img1.dtype
print img2.dtype
img1 *= (1.0/img1.max())
img2 *= (1.0/img2.max())
print img1.max()
print img2.max()

# Add the images together and re-normalize them to have minimum value of 0 and maximum value of 1.
img3 = np.add(img1, img2)
img3 *= (1.0/img3.max())

# Normalize merged result for image display
img3 *= (255.0/img3.max())
cv2.imwrite('image3.jpg', img3)

# Create a new image such that the left half of the image is the left half of img1 and the right half of the image is the right half of img2.
img4 = np.concatenate((img1[:,:(np.size(img1,1)/2)], img2[:,(np.size(img2,1)/2):]), axis=1)

# Normalize merged result for image display
img4 *= (255.0/img4.max())
cv2.imwrite('image4.jpg', img4)






