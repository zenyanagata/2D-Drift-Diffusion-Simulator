import os
import numpy as np
import glob
import matplotlib.pyplot as plt

import constant as C
import parameter as P

def get_filenames(path):
    os.chdir(path)
    filenames = glob.glob("J_*.txt")
    print("{} files were loaded.".format(len(filenames)))
    return filenames


def recutangular_1():
    row = range(72, 98)
    col = range(72, 98)
    region = np.meshgrid(row, col)

    return region


def recutangular_2():
    row = range(62, 108)
    col = range(62, 108)
    region = np.meshgrid(row, col)

    return region

def all(J):
    row = range(0, len(J))
    col = range(0, len(J))
    region = np.meshgrid(row, col)

    return region

def center_line_1():
    row = range(72, 98)
    col = range(72, 98)
    region = np.stack((row, col), axis=1)
    print(region)
    return region

# def center_line_2(width):
#     region = []
#     row = range(72, 98)
#     col = range(72, 98)

#     for w in range(int(width/2)+1):
#         upper_row = [r-w for r in row]
#         upper_col = [c for c in col]
#         lower_row = [r+w for r in row]
#         lower_col = [c for c in col]

#         if w != 0:
#             for i in range(1, w+1):
#                 upper_col.append(upper_col[-1]+i)
#                 upper_row.append(upper_row[-1]+i)

#         temp = np.stack((upper_row, upper_col), axis=1)

#         region.append(temp)

#     print(region)
#     print(np.shape(region))
#     return region

def define_region_plot(is_2D=False, is_1D=True):

    J = np.genfromtxt("J_10_V=0.0005170398202330327.txt")
    region = center_line_1()
    if is_2D:
        for i in region[0]:
            for j in region[1]:
                J[i,j] = np.max(J)
    elif is_1D:
        for r in region:
            i, j = r[0], r[1]
            J[i,j] = np.max(J)
    # region = np.array(region)
    # for w in range(np.shape(region)[0]):
    #     i = region[w, : ,0]
    #     j = region[w,:,1]
    #     J[i,j] = np.max(J)

    plt.figure()
    plt.imshow(J, cmap="rainbow")
    plt.show()


def quantify(filenames, region):
    quantified_J = {}
    region_to_sum = {}
    for filename in filenames:
        J = np.genfromtxt(filename)


def define_region(filenames, path, func, is_2D=False):

    os.chdir(path)

    J = np.genfromtxt(filenames[0])
    region = func()
    if is_2D:
        for i in region[0]:
            for j in region[1]:
                J[i,j] = np.max(J)
    else:
        for r in region:
            i, j = r[0], r[1]
            J[i,j] = np.max(J)

    print("this is the region to sum up J.")
    plt.figure()
    plt.imshow(J, cmap="rainbow")
    plt.show()

    return region


if __name__ == "__main__":

    j_list_rec1 = []
    j_list_rec2 = []
    j_list_all = []
    j_list_line = []
    j_list_surr = []
    vapp_list = []

    v_read = 0.05
    V = v_read / C.VT
    print("V = {}".format(V))



    path = "C:/Users/sakailab/Desktop/nagata/0225/two_elec/cycle1"
    filenames = get_filenames(path)

    for file in filenames:
        j = np.genfromtxt(file)
        print("V = ", file[10: -4])

        vapp_list.append(float(file[10:-4]))


        j_list_rec1.append(np.sum(j[72:98, 72:98].flatten()))
        R_list_rec1 = [V/j_ for j_ in j_list_rec1]

        j_list_rec2.append(np.sum(j[62:108, 62:108].flatten()))
        R_list_rec2 = [V/j_ for j_ in j_list_rec2]

        j_list_all.append(np.sum(j.flatten()))
        R_list_all = [V/j_ for j_ in j_list_all]

        _row = range(72, 98)
        _col = range(72, 98)
        region = np.stack((_row, _col), axis=1)
        sum_j = 0
        for r in region:
                ii, jj = int(r[0]), int(r[1])
                sum_j += j[ii, jj]
        j_list_line.append(sum_j)
        R_list_line = [V/j_ for j_ in j_list_line]

        sum_j = 0
        left_r = range(97, 158)
        for ii in left_r:
            sum_j += j[ii, 97]
        right_r = range(98, 159)
        for ii in right_r:
            sum_j += j[ii, 158]
        up_c = range(98, 159)
        for jj in up_c:
            sum_j += j[97, jj]
        down_c = range(97, 158)
        for jj in up_c:
            sum_j += j[158, jj]
        j_list_surr.append(sum_j)
        R_list_surr = [V/j_ for j_ in j_list_surr]



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
