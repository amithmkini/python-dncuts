import numpy as np
import cv2
from getGaussianAffinity import getGaussianAffinity
from dncuts import dncuts
import matplotlib.pyplot as plt
import imageio
from ncuts import ncuts
import time
import pickle

def visualize(evf, evt, evl, im, nvec):
    vistrue = evt.reshape(len(im[:,0,0]), len(im[0,:,0]), -1)
    visfast = evf.reshape(len(im[:,0,0]), len(im[0,:,0]), -1)
    vist = vistrue[:,:,:nvec]
    visf = visfast[:,:,:nvec]
    
    vistrue = 4 * np.sign(vist) * np.abs(vist)**(1/2)
    visfast = 4 * np.sign(visf) * np.abs(visf)**(1/2)

    vistrue = np.maximum(0, np.minimum(1, vistrue))
    visfast = np.maximum(0, np.minimum(1, visfast))
    g,h,l = vistrue.shape
    m = int(np.floor(np.sqrt(l)))
    n = int(np.ceil(l/m))

    mont_true = np.zeros((g*m, h*n))
    mont_fast = np.zeros((g*m, h*n))

    #Construct montage
    count = 0
    for i in range(m):
        for j in range(n):
            try:
                mont_true[i*g:g+i*g,j*h:h+j*h] = vistrue[:,:,count] 
                mont_fast[i*g:g+i*g,j*h:h+j*h] = visfast[:,:,count]
            except:
                mont_true[i*g:g+i*g,j*h:h+j*h] = 0 
                mont_fast[i*g:g+i*g,j*h:h+j*h] = 0
            count = count + 1
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_title('True eigenvectors: 1 - {0}'.format(nvec))
    ax1.imshow(mont_true)
    ax2 = fig.add_subplot(2,1,2)
    ax2.set_title('Fast eigenvectors: 1 - {0}'.format(nvec))
    ax2.imshow(mont_fast)
    
    plt.savefig('true_vs_fast_montage.png')
    pickle.dump(fig, open('true_vs_fast_montage.pickle', 'wb') )

    #   Plots montage
    # if 'graph' in config and config['graph']:
    #     plt.show();
    plt.clf()

    ax = fig.gca()  
    im = np.zeros((im.shape[0], im.shape[1]))
    ev = visf
    ev = ev.transpose(1,0,2)
    for i in range(nvec):
        im = im + ev[:,:,i]

    im = im.astype(np.float32) 
    ax.imshow(im)
    
    plt.savefig('eigvonimage.png')
    pickle.dump(fig, open('eigvonimage.pickle', 'wb') )

    #   Plots eigenvectors
    # if 'graph' in config and config['graph']:
    #     plt.show();
    plt.clf()


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

visualize(EV_fast, EV_true, EVal_fast, im, NVEC)

# figure; montage(max(0, min(1, vis_true + 0.5))); colormap(betterjet); title('true eigenvectors');
# figure; montage(max(0, min(1, vis_fast + 0.5))); colormap(betterjet); title('fast approximate eigenvectors');

X = im.reshape(im.shape[0]*im.shape[1], im.shape[2])
mu = np.mean(X,0)
Xc = X-mu

im_eig = np.array((EV_fast * (np.matrix(Xc.T)*np.matrix(EV_fast)).T) + mu, dtype='uint8').reshape(im.shape)

# figure; imagesc([im, im_eig]); axis image off;
