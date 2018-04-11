import numpy as np
from ncuts import ncuts
from scipy import sparse
from whiten import whiten

def dncuts(A, NVEC, N_DOWNSAMPLE, DECIMATE, SZ):
    # A = affinity matrix
    # NEVC = number of eigenvectors (set to 16?)
    # N_DOWNSAMPLE = number of downsampling operations (2 seems okay)
    # DECIMATE = amount of decimation for each downsampling operation (set to 2)
    # SZ = size of the image corresponding to A

    A_down = A
    SZ_down = np.array(SZ, dtype=np.int64)
    Bs = {}

    for di in range(N_DOWNSAMPLE):
        # i, j = np.ravel_multi_index(SZ_down, range(A_down.shape[0]))
        (j, i) = np.unravel_index(range(A_down.shape[0]), SZ_down)
        do_keep = np.logical_and((i%DECIMATE == 0),(j%DECIMATE == 0))
        do_keep_idx = np.argwhere(do_keep).flatten()
        A_sub = (A_down[:,do_keep_idx]).T
        d = np.sum(A_sub, 0) + (np.finfo(float).eps)
        B = (A_sub / d).T
        A_down = (A_sub.dot(B)).T

        SZ_down = np.floor(SZ_down / 2)
        SZ_down = np.array(SZ_down, dtype=np.int64)
        Bs[di] = B


    A_down = sparse.csr_matrix(A_down)
    EV, EVal = ncuts(A_down, NVEC)

    for di in range(N_DOWNSAMPLE-1,-1,-1):
        EV = Bs[di] * EV
    
    EVal = (2 ** -N_DOWNSAMPLE) * EVal

    EV = whiten(EV,1, 0)

    return EV, EVal
