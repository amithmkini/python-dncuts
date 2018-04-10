from scipy import sparse
import numpy as np

def ncuts(A, n_ev):
    D = sparse.csr_matrix((np.sum(A.toarray(),0), (range(A.shape[0]), range(A.shape[0]))), shape=(A.shape[0], A.shape[1]))

    nvec = n_ev + 1

    EVal, EV = sparse.linalg.eigs((D - A) + (pow(10,-10) * sparse.eye(D.shape)), M=D, k=nvec, which='sm') # Hoping for the best here
    sortidx = v.argsort()[::-1]
    v = np.diag(EV)
    EVal = v[sortidx[:-1][::-1]]

    EV = EV / np.sqrt(np.sum(EV**2))

    return EV, EVal
