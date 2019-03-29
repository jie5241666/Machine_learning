import numpy as np
a  = np.array([1,2,3,4])
b = np.array([[1,2,3,4],[4,5,6,6]])
print(b.shape)
print(np.sum(b,axis=1).shape)
