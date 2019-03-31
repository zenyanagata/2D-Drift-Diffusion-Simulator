import os
import numpy as np

from initialize import BaseSystem
from poisson_equation import poisson
from continuity_equation import ContinuityEquation
from ion_continuity import Ion
from current_equation import current
import constant as C
import parameter as P
from rendering import render


class System(BaseSystem):
    """
    BaseSystemを継承したクラス
    基本的にこのクラスを各自の目的に合わせて変形していく。

    set_voltage*
        電圧印加プロトコルを定義する。掃引の場合、sigma_a（無次元量, strukov論文参照）, numを引数にして
        電圧(V)を返す。sigma_a == numで1周期。
    
    main_function 
        実行されるシミュレーション。1ループ当たりの印加電圧の増減が大きすぎると、キャリア連続の式が収束しない。
        （expが計算限界より大きい値になってしまう。）推奨はVstep < 0.05。
    """
    def __init__(self, path, new_elecrode_shape=False, electrode=None):
        super().__init__(path=path, new_elecrode_shape=new_elecrode_shape, electrode=electrode)
        self.sigma_a = 0.0
        self.num = 0.01
        

    def set_voltage_sin(self):
        V = 120*np.sin(2*np.pi*(self.sigma_a/P.N**2)/self.num) * C.VT

        return V

    def set_voltage_discrete(self):
        v1 = np.linspace(0, 1, 101)
        v2 = np.linspace(0.99, -1, 201)
        v3 = np.linspace(-0.99, 0, 100)

        v = np.append(v1, np.append(v2, v3))

        index = (self.sigma_a/P.N**2)/self.num // (1/len(v))

        return v[int(index)] * C.VT


    def main_function(self, path_to_save):
        vapp = 0.0
        count_for_render = 0
        jp_list, jn_list = [], []
        vapp_list = []

        carrier = ContinuityEquation(self.boundary, self.dict_a, self.dict_b, self.a0, self.b0, self.row, self.col)

        while (self.sigma_a/P.N**2)/self.num < 1:
            print("(sigma_a/N**2)/num =", (self.sigma_a/P.N**2)/self.num)

            if vapp < 0.2:
                vapp += 0.01
            else:
                vapp = 0.2

            V = self.psi0 + vapp / C.VT  # normalized applied voltage
            print("vapp =", vapp, "V=", V, "psi0=", self.psi0)

            _, self.psi = self.electrode(self.boundary, self.psi, V, self.psi0)  # 印可電圧更新
            ion = Ion(P.h, self.boundary, self.dict_a, self.dict_b, self.a0, self.row, self.col)

            # self-consistent loop
            while True:
                # solve Poisson equation & continuity equations
                self.psi, residue = poisson(P.ETA, self.p, self.n, self.c, self.psi, self.boundary)
                self.p = carrier.continuity(self.psi, self.p0)
                self.n = carrier.continuity(-self.psi, self.n0)
                # converge?
                print("         psi residue:", residue)
                if residue < P.TOLERANCE:
                    break

            self.Nd, self.courant_cond = ion.execute(self.Nd, self.psi, P.Nd0_)
            self.sigma_a += self.courant_cond
            self.c = np.add(self.Nd, -self.Na)



            # evaluate current
            self.jp = current(self.psi, self.p, self.jp)
            self.jp *= -P.UJP
            self.jn = current(-self.psi, self.n, self.jn)
            self.jn *=  P.UJN

            self.j = np.add(self.jp, self.jn)


            jp_list.append(self.jp[int(P.N/2), int(P.N/2)])
            jn_list.append(self.jn[int(P.N/2), int(P.N/2)])
            vapp_list.append(vapp)


            

            if count_for_render % 10 == 0:
                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)
                os.chdir(path_to_save)

                np.savetxt('IV.txt',
                np.column_stack([vapp_list, jp_list, jn_list, np.add(jp_list, jn_list)]), fmt='%15.8e')

                render(path_to_save, self.psi, self.Nd, self.j, self.p, 
                    self.n, self.boundary, count_for_render, vapp)
                

            count_for_render += 1


if __name__ == "__main__":
    from electrode import four_electrode  # 電極形状を指定

    path = "C:/Users/zenya/Documents/研究/simulation/memristor_sim/T1234"  # 電極形状の情報を保管した（する）場所
    system = System(path, new_elecrode_shape=False, electrode=four_electrode)  # Systemクラスのインスタンス生成
    system.main_function(os.path.dirname(os.path.abspath(__file__)) + "/temp")  # main_function(path_to_save)
