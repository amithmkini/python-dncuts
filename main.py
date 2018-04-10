import numpy as np
# from scipy.ndimage import imread
import cv2
from getGaussianAffinity import getGaussianAffinity as GA

r = cv2.imread('lena.bmp', 0)
A = GA(r, 7, 30)
print(A)