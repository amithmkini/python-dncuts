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
EV_true, EVal_true = ncuts(A, NVEC)
end = time.time()
print("Time to generate NCuts: {}".format(end-start))

C = abs(EV_fast.T * EV_true)
M = np.array(range(C.shape[0]))
for pass_ in range(10):
    M_last = M
    for i in range(C.shape[0]):
        for j in range(i+1, C.shape[0]):
            if (C[i,M[j]] + C[j,M[i]]) > (C[i,M[i]] + C[j,M[j]]):
                M[j],M[i] = M[i],M[j]
    if np.array_equal(M, M_last):
        break

junk = (np.array(range(NVEC)) >= M)

EV_fast = EV_fast[:,M]
EV_fast = np.array(EV_fast) * np.array(np.sign(sum(np.multiply(EV_fast, EV_true))))

C = np.matrix(EV_fast.T) * np.matrix(EV_true)
accuracy = np.trace(C)/NVEC

vis_true = np.array(EV_true).reshape(im.shape[0], im.shape[1], 1, 16)
vis_fast = np.array(EV_fast).reshape(im.shape[0], im.shape[1], 1, 16)

vis_true = (4 * np.sign(vis_true)) * np.sqrt(abs(vis_true))
vis_fast = (4 * np.sign(vis_fast)) * np.sqrt(abs(vis_fast))

# figure; montage(max(0, min(1, vis_true + 0.5))); colormap(betterjet); title('true eigenvectors');
# figure; montage(max(0, min(1, vis_fast + 0.5))); colormap(betterjet); title('fast approximate eigenvectors');

X = im.reshape(im.shape[0]*im.shape[1], im.shape[2])
mu = np.mean(X,0)
Xc = X-mu

im_eig = np.array((EV_fast * (np.matrix(Xc.T)*np.matrix(EV_fast)).T) + mu, dtype='uint8').reshape(im.shape)

# figure; imagesc([im, im_eig]); axis image off;
