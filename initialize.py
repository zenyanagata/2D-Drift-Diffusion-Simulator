# 各種変数の初期化・ベースとなるクラスの定義

import numpy as np

import constant as C
import parameter as P
from define_shape import set_ab, set_index, load_ab, load_index

def set_Nd(Nd, boundary, Nd0_):
    m = len(Nd)
    for i in range(0, m):
        for j in range(0, m):
            if boundary[i,j] != 100:
                Nd[i,j] = Nd0_
    return Nd

class BaseSystem():
    def __init__(self, path, new_elecrode_shape=False, electrode=None):
        self.p = np.zeros((P.N+1, P.N+1))            # hole density (/m3)
        self.n = np.zeros((P.N+1, P.N+1))            # electron density (/m3)
        self.Na = np.ones((P.N+1, P.N+1)) * P.Na_    # impurity density Na (/m3)
        self.Nd = np.ones((P.N+1, P.N+1)) * P.Nd_    # impurity density Nd (/m3)
        self.c = np.zeros((P.N+1, P.N+1))            # Net charge from impurity (/m3)
        self.psi = np.zeros((P.N+1, P.N+1))          # potential (V)
        self.dpsi = np.zeros((P.N+1, P.N+1))         # potential update (V)
        self.boundary = np.ones((P.N+1, P.N+1))*100  # shape of electrode
        self.jp = np.zeros((P.N, P.N))               # hole current density (A/m2)
        self.jn = np.zeros((P.N, P.N))               # electron current density (A/m2)
        self.j = np.zeros((P.N, P.N))                # current density (A/m2)

        #initialize boundary, Nd, c
        self.electrode = electrode  # 電極形状を定義した関数
        self.boundary, _ = self.electrode(self.boundary, self.psi, 0, 0)
        self.Nd = set_Nd(self.Nd, self.boundary, P.Nd0_)
        self.c = np.add(self.Nd, -self.Na)
        # initial guess for p & n
        self.p = 0.5 * (np.sqrt(self.c**2 + 4 * C.NI**2) - self.c)
        self.n = 0.5 * (np.sqrt(self.c**2 + 4 * C.NI**2) + self.c)

        # normalization
        self.p *= P.h**3
        self.n *= P.h**3
        self.c *= P.h**3
        self.Nd *= P.h**3
        self.Na *= P.h**3
        self.psi = np.log(self.c / P.NNI)

        # initial guess for normalized potential, carrier (arb constant)
        self.psi0 = self.psi[0,0]
        self.p0, self.n0 = self.p[0,0], self.n[0,0]

        # for continuity equation
        if new_elecrode_shape:
            self.a0, self.b0, self.row, self.col = set_ab(self.boundary)
            self.dict_a, self.dict_b = set_index(self.a0, self.b0, self.row, self.col)
        else:
            self.a0, self.b0, self.row, self.col = load_ab(path)
            self.dict_a, self.dict_b = load_index(path)


    def load(self, path, filename):
        self.p = np.genfromtxt(path + '\\' + filename["p"]) * P.h**3
        self.n = np.genfromtxt(path + filename["n"]) * P.h**3
        self.Nd = np.genfromtxt(path + filename["Nd"]) * P.h**3
        self.c = np.add(self.Nd, -self.Na)


    def render(self):
        print("render関数を定義してください")

    def main(self):
        # main loop
        print("main関数を定義してください")
    
    
