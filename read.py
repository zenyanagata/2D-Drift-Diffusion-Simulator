import os
import glob
import numpy as np

from initialize import BaseSystem
from poisson_equation import poisson
from continuity_equation import ContinuityEquation
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



    def main_function(self, path_to_save):
        j_list_rec1 = []
        j_list_rec2 = []
        j_list_all = []
        j_list_line = []
        j_list_surr = []
        vapp_list = []

        v_read = 0.05
        V = self.psi0 + v_read / C.VT  # normalized applied voltage

        print("  vapp =", v_read, "V=", V, "psi0=", self.psi0)

        carrier = ContinuityEquation(self.boundary, self.dict_a, self.dict_b, self.a0, self.b0, self.row, self.col)
        os.chdir("C:/Users/sakailab/Desktop/nagata/0225/two_elec/cycle1")
        filenames = glob.glob("Nd_*.txt")
        print("{} files were found.".format(len(filenames)))
        for name in filenames:
            print("loading Nd from {}.".format(name))

            filename = {"Nd": name}
            self.load("C:/Users/sakailab/Desktop/nagata/0225/two_elec/cycle1/", filename, eraseT24=False)

            if name[3:8].isdigit():
                count = int(name[3:8])
            else:
                print("count not understood.")
            print("file number: {}".format(count))


            _, self.psi = self.electrode(self.boundary, self.psi, V, self.psi0)  # 印可電圧更新

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

            # evaluate current
            self.jp = current(self.psi, self.p, self.jp)
            self.jp *= -P.UJP
            self.jn = current(-self.psi, self.n, self.jn)
            self.jn *=  P.UJN

            self.j = np.add(self.jp, self.jn)


            vapp_list.append(float(name[11:-4]))


            j_list_rec1.append(np.sum(self.j[72:98, 72:98].flatten()))
            R_list_rec1 = [V/j_ for j_ in j_list_rec1]

            j_list_rec2.append(np.sum(self.j[62:108, 62:108].flatten()))
            R_list_rec2 = [V/j_ for j_ in j_list_rec2]

            j_list_all.append(np.sum(self.j.flatten()))
            R_list_all = [V/j_ for j_ in j_list_all]

            _row = range(72, 98)
            _col = range(72, 98)
            region = np.stack((_row, _col), axis=1)
            sum_j = 0
            for r in region:
                    ii, jj = int(r[0]), int(r[1])
                    sum_j += self.j[ii, jj]
            j_list_line.append(sum_j)
            R_list_line = [V/j_ for j_ in j_list_line]

            sum_j = 0
            left_r = range(97, 158)
            for ii in left_r:
                sum_j += self.j[ii, 97]
            right_r = range(98, 159)
            for ii in right_r:
                sum_j += self.j[ii, 158]
            up_c = range(98, 159)
            for jj in up_c:
                sum_j += self.j[97, jj]
            down_c = range(97, 158)
            for jj in up_c:
                sum_j += self.j[158, jj]
            j_list_surr.append(sum_j)
            R_list_surr = [V/j_ for j_ in j_list_surr]


            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            os.chdir(path_to_save)


            render(path_to_save, self.psi, self.Nd, self.j, self.p, self.n, self.boundary, count, v_read)

        np.savetxt('IV_rec1.txt',
            np.column_stack([vapp_list, j_list_rec1, R_list_rec1]), fmt='%15.8e')

        np.savetxt('IV_rec2.txt',
            np.column_stack([vapp_list, j_list_rec2, R_list_rec2]), fmt='%15.8e')

        np.savetxt('IV_all.txt',
            np.column_stack([vapp_list, j_list_all, R_list_all]), fmt='%15.8e')

        np.savetxt('IV_line.txt',
            np.column_stack([vapp_list, j_list_line, R_list_line]), fmt='%15.8e')

        np.savetxt('IV_surr.txt',
            np.column_stack([vapp_list, j_list_surr, R_list_surr]), fmt='%15.8e')

        np.savetxt('Vapp.txt', vapp_list, fmt='%15.8e')




if __name__ == "__main__":
    from electrode import two_electrode_13  # 電極形状を指定

    path = "C:/Users/sakailab/Desktop/nagata/memristor_sim/T13"  # 電極形状の情報を保管した（する）場所
    system = System(path, new_elecrode_shape=False, electrode=two_electrode_13)  # Systemクラスのインスタンス生成
    system.main_function("C:/Users/sakailab/Desktop/nagata/0225/two_elec/read_cycle1")  # main_function(path_to_save)
