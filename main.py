import numpy as np
from scipy import sparse
# from scipy.ndimage import imread
import cv2
from getGaussianAffinity import getGaussianAffinity as GA
from dncuts import dncuts

A = np.random.rand(100,100)
B = sparse.csr_matrix((np.sum(A,0), (range(A.shape[0]), range(A.shape[0]))), shape=(A.shape[0], A.shape[1]))

EV, EVal = dncuts(B, 16, 2, 2, B.shape)
