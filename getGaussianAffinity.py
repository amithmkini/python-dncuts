import numpy as np
import numpy.matlib
from scipy import sparse

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
#     ind[ind < 0] = 0
#     ind[ind >= array_shape[0]*array_shape[1]] = 0
    return ind

# def sub2ind(shape, row, col):
#     cols = shape[1]
#     return row*cols+col

def getGaussianAffinity(im, xy_radius, rgb_sigma):
    sz = [im.shape[0], im.shape[1]]
    
    # Find all pairs of pixels within a distance of XY_RADIUS
    x = list(range(-xy_radius, xy_radius+1, 1))
    y = list(range(-xy_radius, xy_radius+1, 1))
    dj, di = np.meshgrid(x, y)
    dv = (np.square(dj)+np.square(di)<=xy_radius**2)
    
    di = di[dv]
    dj = dj[dv]
    

    x = list(range(0, im.shape[0]))
    y = list(range(0, im.shape[1]))
    j, i = np.meshgrid(x, y)
    
    
    i = np.tile(i.flatten().reshape(-1,1), (1, len(di)))
    j = np.tile(j.flatten().reshape(-1,1), (1, len(dj)))
    
    
    i_ = i + di.T
    j_ = j + di.T
    
    
    v = (i_>=0)&(i_<im.shape[0])&(j_>=0)&(j_<im.shape[1])
    # rows+(cols-1)*size(M, 1)
    pair_i = sub2ind(sz, i[v], j[v])
    pair_j = sub2ind(sz, i_[v], j_[v])
    
    # Weight each pair by the difference in RGB values, divided by RGB_SIGMA
    RGB = np.reshape(im, (im.shape[0]*im.shape[1],-1)).astype('float32')
    RGB = RGB/rgb_sigma
    
    W = np.exp(-np.sum(np.square(RGB[pair_i,:] - RGB[pair_j,:]),axis=1))
    # Construct an affinity matrix
    A = sparse.csr_matrix((W, (pair_i, pair_j)), shape=(np.prod(sz, axis=0), np.prod(sz, axis=0)))
    return A

