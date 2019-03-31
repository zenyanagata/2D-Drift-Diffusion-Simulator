# 定常状態でのキャリア連続の式

import numpy as np
from scipy.sparse.linalg import spsolve, lsqr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import dsolve
from numba.decorators import jit
import warnings

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


warnings.filterwarnings("error")

class ContinuityEquation():
    def __init__(self, boundary, dict_a, dict_b, a0, b0, row, col):
        self.boundary = boundary
        self.a_2i, self.a_2j = dict_a["a_2i"], dict_a["a_2j"]
        self.a_3i, self.a_3j = dict_a["a_3i"], dict_a["a_3j"]
        self.a_4i, self.a_4j = dict_a["a_4i"], dict_a["a_4j"]
        self.a_5i, self.a_5j = dict_a["a_5i"], dict_a["a_5j"]
        self.a_6i, self.a_6j = dict_a["a_6i"], dict_a["a_6j"]
        self.a_7i, self.a_7j = dict_a["a_7i"], dict_a["a_7j"]
        self.a_8i, self.a_8j = dict_a["a_8i"], dict_a["a_8j"]
        self.a_9i, self.a_9j = dict_a["a_9i"], dict_a["a_9j"]
        self.a_11i, self.a_11j = dict_a["a_11i"], dict_a["a_11j"]
        self.a_1i, self.a_1j = dict_a["a_1i"], dict_a["a_1j"]
        self.a_10i, self.a_10j = dict_a["a_10i"], dict_a["a_10j"]
        self.a_100i, self.a_100j = dict_a["a_100i"], dict_a["a_100j"]
        self.a_1000i, self.a_1000j = dict_a["a_1000i"], dict_a["a_1000j"]
        self.b_1 = dict_b["b_1"]
        self.b_10 = dict_b["b_10"]
        self.b_100 = dict_b["b_100"]
        self.b_1000 = dict_b["b_1000"]
        self.b_101 = dict_b["b_101"]
        self.b_1001 = dict_b["b_1001"]
        self.b_110 = dict_b["b_110"]
        self.b_1010 = dict_b["b_1010"]
        self.a0 = a0
        self.b0 = b0
        self.row = row
        self.col = col

    def continuity(self, psi, pn0):
        a = np.zeros_like(self.a0, dtype=np.float32)
        b = np.zeros_like(self.b0, dtype=np.float32)
        m = len(psi)

        b[self.b_1]    = - bernoulli(psi[self.row[self.b_1], self.col[self.b_1]] - psi[self.row[self.b_1], self.col[self.b_1]+1]) * pn0               # 右
        b[self.b_10]   = - bernoulli(psi[self.row[self.b_10], self.col[self.b_10]] - psi[self.row[self.b_10], self.col[self.b_10]-1]) * pn0           # 左
        b[self.b_100]  = - bernoulli(psi[self.row[self.b_100], self.col[self.b_100]] - psi[self.row[self.b_100]-1, self.col[self.b_100]]) * pn0       # 上
        b[self.b_1000] = - bernoulli(psi[self.row[self.b_1000], self.col[self.b_1000]] - psi[self.row[self.b_1000]+1, self.col[self.b_1000]]) * pn0   # 下
        b[self.b_101]  = - bernoulli(psi[self.row[self.b_101], self.col[self.b_101]] - psi[self.row[self.b_101], self.col[self.b_101]+1]) * pn0 \
                    - bernoulli(psi[self.row[self.b_101], self.col[self.b_101]] - psi[self.row[self.b_101]-1, self.col[self.b_101]]) * pn0            # 右上
        b[self.b_1001] = - bernoulli(psi[self.row[self.b_1001], self.col[self.b_1001]] - psi[self.row[self.b_1001], self.col[self.b_1001]+1]) * pn0 \
                    - bernoulli(psi[self.row[self.b_1001], self.col[self.b_1001]] - psi[self.row[self.b_1001]+1, self.col[self.b_1001]]) * pn0        # 右下
        b[self.b_110]  = - bernoulli(psi[self.row[self.b_110], self.col[self.b_110]] - psi[self.row[self.b_110], self.col[self.b_110]-1]) * pn0 \
                    - bernoulli(psi[self.row[self.b_110], self.col[self.b_110]] - psi[self.row[self.b_110]-1, self.col[self.b_110]]) * pn0            # 左上
        b[self.b_1010] = - bernoulli(psi[self.row[self.b_1010], self.col[self.b_1010]] - psi[self.row[self.b_1010], self.col[self.b_1010]-1]) * pn0 \
                    - bernoulli(psi[self.row[self.b_1010], self.col[self.b_1010]] - psi[self.row[self.b_1010]+1, self.col[self.b_1010]]) * pn0        # 左下


        a[self.a_2i, self.a_2j] = - bernoulli(psi[(self.row[self.a_2i]+1), self.col[self.a_2i]] - psi[self.row[self.a_2i], self.col[self.a_2i]]) \
                        - bernoulli(psi[(self.row[self.a_2i]-1), self.col[self.a_2i]] - psi[self.row[self.a_2i], self.col[self.a_2i]]) \
                        - bernoulli(psi[self.row[self.a_2i], (self.col[self.a_2i]+1)] - psi[self.row[self.a_2i], self.col[self.a_2i]]) \
                        - bernoulli(psi[self.row[self.a_2i], (self.col[self.a_2i]-1)] - psi[self.row[self.a_2i], self.col[self.a_2i]])

        a[self.a_3i, self.a_3j] = - bernoulli(psi[(self.row[self.a_3i]+1), self.col[self.a_3i]] - psi[self.row[self.a_3i], self.col[self.a_3i]]) \
                        - bernoulli(psi[self.row[self.a_3i], (self.col[self.a_3i]+1)] - psi[self.row[self.a_3i], self.col[self.a_3i]]) \
                        - bernoulli(psi[self.row[self.a_3i], (self.col[self.a_3i]-1)] - psi[self.row[self.a_3i], self.col[self.a_3i]])

        a[self.a_4i, self.a_4j] = - bernoulli(psi[(self.row[self.a_4i]-1), self.col[self.a_4i]] - psi[self.row[self.a_4i], self.col[self.a_4i]]) \
                        - bernoulli(psi[self.row[self.a_4i], (self.col[self.a_4i]+1)] - psi[self.row[self.a_4i], self.col[self.a_4i]]) \
                        - bernoulli(psi[self.row[self.a_4i], (self.col[self.a_4i]-1)] - psi[self.row[self.a_4i], self.col[self.a_4i]])

        a[self.a_5i, self.a_5j] = - bernoulli(psi[(self.row[self.a_5i]+1), self.col[self.a_5i]] - psi[self.row[self.a_5i], self.col[self.a_5i]]) \
                        - bernoulli(psi[(self.row[self.a_5i]-1), self.col[self.a_5i]] - psi[self.row[self.a_5i], self.col[self.a_5i]]) \
                        - bernoulli(psi[self.row[self.a_5i], (self.col[self.a_5i]+1)] - psi[self.row[self.a_5i], self.col[self.a_5i]])

        a[self.a_6i, self.a_6j] = - bernoulli(psi[(self.row[self.a_6i]+1), self.col[self.a_6i]] - psi[self.row[self.a_6i], self.col[self.a_6i]]) \
                        - bernoulli(psi[(self.row[self.a_6i]-1), self.col[self.a_6i]] - psi[self.row[self.a_6i], self.col[self.a_6i]]) \
                        - bernoulli(psi[self.row[self.a_6i], (self.col[self.a_6i]-1)] - psi[self.row[self.a_6i], self.col[self.a_6i]])

        a[self.a_7i, self.a_7j] = - bernoulli(psi[(self.row[self.a_7i]+1), self.col[self.a_7i]] - psi[self.row[self.a_7i], self.col[self.a_7i]]) \
                        - bernoulli(psi[self.row[self.a_7i], (self.col[self.a_7i]+1)] - psi[self.row[self.a_7i], self.col[self.a_7i]])

        a[self.a_8i, self.a_8j] = - bernoulli(psi[(self.row[self.a_8i]+1), self.col[self.a_8i]] - psi[self.row[self.a_8i], self.col[self.a_8i]]) \
                        - bernoulli(psi[self.row[self.a_8i], (self.col[self.a_8i]-1)] - psi[self.row[self.a_8i], self.col[self.a_8i]])

        a[self.a_9i, self.a_9j] = - bernoulli(psi[(self.row[self.a_9i]-1), self.col[self.a_9i]] - psi[self.row[self.a_9i], self.col[self.a_9i]]) \
                        - bernoulli(psi[self.row[self.a_9i], (self.col[self.a_9i]+1)] - psi[self.row[self.a_9i], self.col[self.a_9i]])

        a[self.a_11i, self.a_11j] = - bernoulli(psi[(self.row[self.a_11i]-1), self.col[self.a_11i]] - psi[self.row[self.a_11i], self.col[self.a_11i]]) \
                        - bernoulli(psi[self.row[self.a_11i], (self.col[self.a_11i]-1)] - psi[self.row[self.a_11i], self.col[self.a_11i]])

        a[self.a_1i, self.a_1j] = bernoulli(psi[self.row[self.a_1i], self.col[self.a_1i]] - psi[self.row[self.a_1i], self.col[self.a_1i]+1])

        a[self.a_10i, self.a_10j] = bernoulli(psi[self.row[self.a_10i], self.col[self.a_10i]] - psi[self.row[self.a_10i], self.col[self.a_10i]-1])

        a[self.a_100i, self.a_100j] = bernoulli(psi[self.row[self.a_100i], self.col[self.a_100i]] - psi[self.row[self.a_100i]-1, self.col[self.a_100i]])

        a[self.a_1000i, self.a_1000j] = bernoulli(psi[self.row[self.a_1000i], self.col[self.a_1000i]] - psi[self.row[self.a_1000i]+1, self.col[self.a_1000i]])


        # solve a * pn = b for pn
        try:
            a = csr_matrix(a)
            pn = spsolve(a, b, use_umfpack=True)
        except dsolve.linsolve.MatrixRankWarning:
            pn = lsqr(a, b)[0]

        boundary_ = np.reshape(self.boundary, (m*m))
        for i in range(m*m):
            if boundary_[i] != 100:
                pn = np.insert(pn, i, pn0, axis=0)

        pn = np.reshape(pn, (m, m))

        return pn
