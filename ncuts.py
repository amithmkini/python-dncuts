from scipy import sparse
from scipy.sparse import linalg
import numpy as np

def ncuts(A, n_ev):

    Asum = np.sum(A,0)
    Asum_new = np.copy(Asum).reshape((-1,))
    D = sparse.csc_matrix((Asum_new, (range(A.shape[0]), range(A.shape[0]))), shape=(A.shape[0], A.shape[1]))
    nvec = n_ev
    A_input = (D - A) + (pow(10,-10) * sparse.eye(D.shape[0]))
    D = D.astype(np.float64)

    EVal, EV = linalg.eigs(A_input, M=D, k=nvec, which='SM', tol=1, sigma=0.1)
    v = np.diag(EV)
    sortidx = v.argsort()[::-1]
    EVal = v[sortidx[:-1][::-1]]

    EV = EV / np.sqrt(np.sum(EV**2))

    return EV, EVal
