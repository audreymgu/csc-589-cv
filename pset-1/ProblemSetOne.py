import numpy as np

# Define Matrix M and Vectors a, b, c.

# Matrices
M = np.array([(1,2,3),(4,5,6),(7,8,9),(0,2,2)])

# Vectors
a = np.array([1,1,0])
a.shape = (3,1)
print a.ndim

b = np.array([-1,2,5])
b.shape = (3,1)
print b.ndim

c = np.array([0,2,3,2])
c.shape = (4,1)
print c.ndim

# a = np.array([[1],[1],[0]])
# b = np.array([[-1],[2],[5]])
# c = np.array([[0],[2],[3],[2]])

# a = np.empty((3,1))
# a[0,0] = 1
# a[1,0] = 1
# a[2,0] = 0

# b = np.empty((3,1))
# b[0,0] = -1
# b[1,0] = 2
# b[2,0] = 5

# c = np.empty((4,1))
# c[0,0] = 0
# c[1,0] = 2
# c[2,0] = 3
# c[3,0] = 2

# Find the dot product (also known as the inner product) of vectors a and b. Save the resulting value as variable 'aDotb'.
aDotb = np.inner(a,b)
print aDotb

# Find the element-wise product of a and b.
aElementb = np.multiply(a,b)
print aElementb

# Find (a^Tb)^Ma.
aMatrixb = np.outer(a,b)
aMatrixb = aMatrixb.T
print aMatrixb

# Without using a loop, multiply each row and of M element-wise by a.
