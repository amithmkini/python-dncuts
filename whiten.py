import numpy as np

with open('evarraygg','r') as f:
    x = f.read()
x = x.split('\n')
x = x[:len(x)-1]
x = np.matrix([[float(__) for __ in _.split(',')] for _ in x])

def whiten(X, DO_CENTER = 1, V_PAD = 0.1):
    N_cov = 100000
    m = np.mean(X, 0)
    if DO_CENTER == 0:
        m = 0*m
    X_zeroed = X - np.matrix(np.ones((X.shape[0],1))).dot(m)
    np.random.seed(0)
    rows = np.random.rand(X_zeroed.shape[0],1) <= N_cov/X_zeroed.shape[0]
    X_sub = X_zeroed[rows.reshape(rows.shape[0],),:]
    C = (X_sub.T * X_sub) / X_sub.shape[0]
    D,V = np.linalg.eig(C)

    iD = np.matrix(
            np.diag(
                np.sqrt(1/(D + V_PAD))
            )
        )
    map_ = V * iD * V.T

    X_white = X_zeroed * map_

    mag = np.sqrt(np.mean(np.array(X_white.flatten()) ** 2))
    X_white = X_white/mag

    return X_white

print(whiten(x, 1, 0))
