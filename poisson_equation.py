#ポアソン方程式

import numpy as np
from scipy.sparse.linalg import spsolve, lsqr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import dsolve
import constant as C

def poisson(eta, p, n, c, psi, boundary):
    '''
    Solve Poisson equation
    '''
    m = len(p) #number od mesh N+1

    a = np.zeros((m*m, m*m), dtype=np.float32)
    b = np.zeros(m*m, dtype=np.float32)

    i = np.arange(1, m-1) # 1 ~ N-1
    for j in range(1, m-1):
        # diagonal elements
        a[i*m + j, i*m + j] = 4.0 + eta * (p[i, j] + n[i, j])
        # non-diagonal elements
        a[i*m + j, i*m + (j+1)] = -1  #dpsi[i,j+1]
        a[i*m + j, i*m + (j-1)] = -1  #dpsi[i,j-1]
        a[i*m + j, (i+1)*m + j] = -1  #dpsi[i+1,j]
        a[i*m + j, (i-1)*m + j] = -1  #dpsi[i-1,j]
        # right-hand side
        b[i*m + j] = eta * (p[i, j] - n[i, j] + c[i, j]) \
                               + psi[i, (j-1)] + psi[i, (j+1)] \
                               + psi[(i-1), j] + psi[(i+1), j] \
                               - 4.0 * psi[i, j]
    # 界面　upper side
    i = 0
    j = np.arange(1, m-1)
    # diagonal elements
    a[i*m + j, i*m + j] = 3.0 + eta * (p[i, j] + n[i, j])
    # non-diagonal elements
    a[i*m + j, i*m + (j+1)] = -1  #dpsi[i,j+1]
    a[i*m + j, i*m + (j-1)] = -1  #dpsi[i,j-1]
    a[i*m + j, (i+1)*m + j] = -1  #dpsi[i+1,j]
    # right-hand side
    b[i*m + j] = eta * (p[i, j] - n[i, j] + c[i, j]) \
                            + psi[i, (j-1)] + psi[i, (j+1)] \
                            + psi[(i+1), j] \
                            - 3.0 * psi[i, j]
    # 界面　lower side
    i = m-1  # N
    # diagonal elements
    a[i*m + j, i*m + j] = 3.0 + eta * (p[i, j] + n[i, j])
    # non-diagonal elements
    a[i*m + j, i*m + (j+1)] = -1  #dpsi[i,j+1]
    a[i*m + j, i*m + (j-1)] = -1  #dpsi[i,j-1]
    a[i*m + j, (i-1)*m + j] = -1  #dpsi[i-1,j]
    # right-hand side
    b[i*m + j] = eta * (p[i, j] - n[i, j] + c[i, j]) \
                            + psi[i, (j-1)] + psi[i, (j+1)] \
                            + psi[(i-1), j] \
                            - 3.0 * psi[i, j]
    # 界面　left side
    i = np.arange(1, m-1)
    j = 0
    # diagonal elements
    a[i*m + j, i*m + j] = 3.0 + eta * (p[i, j] + n[i, j])
    # non-diagonal elements
    a[i*m + j, i*m + (j+1)] = -1  #dpsi[i,j+1]
    a[i*m + j, (i+1)*m + j] = -1  #dpsi[i+1,j]
    a[i*m + j, (i-1)*m + j] = -1  #dpsi[i-1,j]
    # right-hand side
    b[i*m + j] = eta * (p[i, j] - n[i, j] + c[i, j]) \
                            + psi[i, (j+1)] \
                            + psi[(i-1), j] + psi[(i+1), j] \
                            - 3.0 * psi[i, j]
    # 界面　right side
    j = m-1
    # diagonal elements
    a[i*m + j, i*m + j] = 3.0 + eta * (p[i, j] + n[i, j])
    # non-diagonal elements
    a[i*m + j, i*m + (j-1)] = -1  #dpsi[i,j-1]
    a[i*m + j, (i+1)*m + j] = -1  #dpsi[i+1,j]
    a[i*m + j, (i-1)*m + j] = -1  #dpsi[i-1,j]
    # right-hand side
    b[i*m + j] = eta * (p[i, j] - n[i, j] + c[i, j]) \
                            + psi[i, (j-1)] \
                            + psi[(i-1), j] + psi[(i+1), j] \
                            - 3.0 * psi[i, j]
    # 界面　upper left
    i = 0
    j = 0
    # diagonal elements
    a[i*m + j, i*m + j] = 2.0 + eta * (p[i, j] + n[i, j])
    # non-diagonal elements
    a[i*m + j, i*m + (j+1)] = -1  #dpsi[i,j+1]
    a[i*m + j, (i+1)*m + j] = -1  #dpsi[i+1,j]
    # right-hand side
    b[i*m + j] = eta * (p[i, j] - n[i, j] + c[i, j]) \
                            + psi[i, (j+1)] \
                            + psi[(i+1), j] \
                            - 2.0 * psi[i, j]
    # 界面　upper right
    i = 0
    j = m-1
    # diagonal elements
    a[i*m + j, i*m + j] = 2.0 + eta * (p[i, j] + n[i, j])
    # non-diagonal elements
    a[i*m + j, i*m + (j-1)] = -1  #dpsi[i,j-1]
    a[i*m + j, (i+1)*m + j] = -1  #dpsi[i+1,j]
    # right-hand side
    b[i*m + j] = eta * (p[i, j] - n[i, j] + c[i, j]) \
                            + psi[i, (j-1)] \
                            + psi[(i+1), j] \
                            - 2.0 * psi[i, j]
    # 界面　lower left
    i = m-1
    j = 0
    # diagonal elements
    a[i*m + j, i*m + j] = 2.0 + eta * (p[i, j] + n[i, j])
    # non-diagonal elements
    a[i*m + j, i*m + (j+1)] = -1  #dpsi[i,j+1]
    a[i*m + j, (i-1)*m + j] = -1  #dpsi[i-1,j]
    # right-hand side
    b[i*m + j] = eta * (p[i, j] - n[i, j] + c[i, j]) \
                            + psi[i, (j+1)] \
                            + psi[(i-1), j] \
                            - 2.0 * psi[i, j]
    # 界面　lower right
    i = m-1
    j = m-1
    # diagonal elements
    a[i*m + j, i*m + j] = 2.0 + eta * (p[i, j] + n[i, j])
    # non-diagonal elements
    a[i*m + j, i*m + (j-1)] = -1  #dpsi[i,j-1]
    a[i*m + j, (i-1)*m + j] = -1  #dpsi[i-1,j]
    # right-hand side
    b[i*m + j] = eta * (p[i, j] - n[i, j] + c[i, j]) \
                            + psi[i, (j-1)] \
                            + psi[(i-1), j] \
                            - 2.0 * psi[i, j]

    # solve a * dpsi = b for dpsi
    a = csr_matrix(a)
    dpsi = spsolve(a, b, use_umfpack=True)

    #電極部の電位変化はゼロ
    for i in range(0, m):
        for j in range(0, m):
            if boundary[i, j] != 100:  # if electrode, no change in psi.
                dpsi[i*m + j] = 0

    residue = np.sqrt(np.sum(dpsi**2) / len(dpsi)) * C.VT # (V)

    dpsi = np.reshape(dpsi, (m, m))
    psi += dpsi

    return psi, residue