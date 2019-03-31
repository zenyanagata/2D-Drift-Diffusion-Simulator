# 電気伝導の式

import numpy as np
from numba.decorators import jit


@jit
def bernoulli_(x):
    '''
    Bernoulli function
    '''
    if np.abs(x) < 1e-2:
        return 1.0 - x * (0.5 - x * (1.0 / 12 - x**2 / 720))
    else:
        return x / (np.exp(x) - 1.0)

bernoulli = np.frompyfunc(bernoulli_, 1, 1) # universal function


def current(psi, pn, jpn):
    '''
    Evaluate current
    '''
    jpn_x = np.zeros_like(jpn)
    jpn_y = np.zeros_like(jpn)
    m = len(psi) - 1 # N
    i = np.arange(m)
    for j in range(m):
        jpn_x[i,j] =   bernoulli(psi[i,j] - psi[i,j+1]) * pn[i,j+1] \
                     - bernoulli(psi[i,j+1] - psi[i,j]) * pn[i,j]
        jpn_y[i,j] =   bernoulli(psi[i,j] - psi[i+1,j]) * pn[i+1,j] \
                     - bernoulli(psi[i+1,j] - psi[i,j]) * pn[i,j]

    jpn = np.sqrt(jpn_x**2 + jpn_y**2)

    return jpn