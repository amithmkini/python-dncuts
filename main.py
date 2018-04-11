import numpy as np
import cv2
from getGaussianAffinity import getGaussianAffinity
from dncuts import dncuts
from ncuts import ncuts
import time

XY_RADIUS = 7
RGB_SIGMA = 30
NVEC = 16

DNCUTS_N_DOWNSAMPLE = 2
DNCUTS_DECIMATE = 2


start = time.time()
im = cv2.resize(cv2.imread('lena.bmp'), (128, 128))
sz = im.shape
end = time.time()
print("Time to load image: {}".format(end-start))

start = time.time()
A = getGaussianAffinity(im, XY_RADIUS, RGB_SIGMA)
end = time.time()
print("Time to construct affinity: {}".format(end - start))

start = time.time()
EV_fast, EVal_fast = dncuts(A, NVEC, DNCUTS_N_DOWNSAMPLE, DNCUTS_DECIMATE, sz)
end = time.time()
print("Time to generate DNCuts: {}".format(end-start))

start = time.time()
EV, EVal = ncuts(A, NVEC)
end = time.time()
print("Time to generate NCuts: {}".format(end-start))
