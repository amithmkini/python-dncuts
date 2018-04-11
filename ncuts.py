from scipy import sparse
from scipy.sparse import linalg
import numpy as np

def ncuts(A, n_ev):

    Asum = np.sum(A.toarray(),0).flatten()

    D = sparse.csr_matrix((Asum, (range(A.shape[0]), range(A.shape[0]))), shape=(A.shape[0], A.shape[1]))

    nvec = n_ev + 1
    EVal, EV = linalg.eigs((D - A) + (pow(10,-10) * sparse.eye(D.shape[0])), M=D, k=nvec, which='SM')
    v = np.diag(EV)
    sortidx = v.argsort()[::-1]
    EVal = v[sortidx[:-1][::-1]]

    EV = EV / np.sqrt(np.sum(EV**2))

    return EV, EVal
