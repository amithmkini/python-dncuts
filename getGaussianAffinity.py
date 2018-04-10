import numpy as np
import scipy import sparse

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def getGaussianAffinity(im, xy_radius, rgb_sigma):
    sz = [im.shape[0], im.shape[1]]
    
    # Find all pairs of pixels within a distance of XY_RADIUS
    x = list(range(-xy_radius, xy_radius+1, 1))
    y = list(range(-xy_radius, xy_radius+1, 1))
    dj, di = np.meshgrid(x, y)
    dv = (np.square(dj)+np.square(di)<=xy_radius)
    di = di[dv]
    dj = dj[dv]

    x = list(range(1, im.shape[0]+1))
    y = list(range(1, im.shape[1]+1))
    j, i = np.meshgrid(x, y)
    
    i = np.tile(i, (1, len(di)))
    j = np.tile(j, (1, len(di)))
    
    i_ = np.add(i, di.T)
    j_ = np.add(j, dj.T)
    
    v = (i_>=1)&(i_<=im.shape[0])&(j_>=1)&(j_<=im.shape[1])
    pair_i = sub2ind(sz, i[v], j[v]);
    pair_j = sub2ind(sz, i_[v], j_[v]);
    
    # Weight each pair by the difference in RGB values, divided by RGB_SIGMA
    RGB = np.reshape(im, -1, im.shape[2]).astype('float32')
    RGB = RGB/rgb_sigma
    W = np.exp(-np.sum(np.square(RGB[pair_i,:] - RGB[pair_j,:]),axis=1))
    
    # Construct an affinity matrix
    A = sparse.csr_matrix((pair_i, pair_j, W), shape=(np.prod(sz, axis=0), np.prod(sz, axis=0)))
    return A
