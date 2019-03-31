# 連続の式で用いる電極形状の情報

import numpy as np
import os

def set_ab(boundary):
    print("Setting ab")
    m = len(boundary)  # N+1
    a = np.zeros((m*m, m*m), dtype=np.int16)
    b = np.zeros(m*m, dtype=np.int16)
    # diagonal elements
    i = np.arange(1, m-1) # 1 ~ N-1
    for j in range(1, m-1):
        a[i*m + j, i*m + j] = 2
    # sub-diagonal elements
        a[i*m + j, i*m + (j+1)] = 1 # 右
        a[i*m + j, i*m + (j-1)] = 10 # 左
        a[i*m + j, (i+1)*m + j] = 10**3 # 下
        a[i*m + j, (i-1)*m + j] = 10**2 # 上

    i = 0
    j = np.arange(1, m-1) # 1 ~ N-1
    a[i*m + j, i*m + j] = 3
    # sub-diagonal elements
    a[i*m + j, i*m + (j+1)] = 1 # 右
    a[i*m + j, i*m + (j-1)] = 10 # 左
    a[i*m + j, (i+1)*m + j] = 10**3 # 下

    i = m-1 # N
    j = np.arange(1, m-1) # 1 ~ N-1
    a[i*m + j, i*m + j] = 4
    # sub-diagonal elements
    a[i*m + j, i*m + (j+1)] = 1 # 右
    a[i*m + j, i*m + (j-1)] = 10 # 左
    a[i*m + j, (i-1)*m + j] = 10**2 # 上

    i = np.arange(1, m-1) # 1 ~ N-1
    j = 0
    a[i*m + j, i*m + j] = 5
    # sub-diagonal elements
    a[i*m + j, i*m + (j+1)] = 1 # 右
    a[i*m + j, (i+1)*m + j] = 10**3 # 下
    a[i*m + j, (i-1)*m + j] = 10**2 # 上

    i = np.arange(1, m-1) # 1 ~ N-1
    j = m-1  # N
    a[i*m + j, i*m + j] = 6
    # sub-diagonal elements
    a[i*m + j, i*m + (j-1)] = 10 # 左
    a[i*m + j, (i+1)*m + j] = 10**3 # 下
    a[i*m + j, (i-1)*m + j] = 10**2 # 上

    i, j = 0, 0
    a[i*m + j, i*m + j] = 7
    # sub-diagonal elements
    a[i*m + j, i*m + (j+1)] = 1 # 右
    a[i*m + j, (i+1)*m + j] = 10**3 # 下

    i, j = 0, m-1
    a[i*m + j, i*m + j] = 8
    # sub-diagonal elements
    a[i*m + j, i*m + (j-1)] = 10 # 左
    a[i*m + j, (i+1)*m + j] = 10**3 # 下

    i, j = m-1, 0
    a[i*m + j, i*m + j] = 9
    # sub-diagonal elements
    a[i*m + j, i*m + (j+1)] = 1 # 右
    a[i*m + j, (i-1)*m + j] = 10**2 # 上


    i,j = m-1, m-1
    a[i*m + j, i*m + j] = 11
    # sub-diagonal elements
    a[i*m + j, i*m + (j-1)] = 10 # 左
    a[i*m + j, (i-1)*m + j] = 10**2 # 上


    row = []
    for i in range(m):
        for j in range(m):
            row.append(i)
    row = np.array(row)
    col = np.array([i for i in range(m)]*m)

    for ii in range(m):
        for jj in range(m):
            if boundary[ii, jj] != 100:
                column = a[:, ii*m + jj]
                for i in range(m*m):
                    if column[i] != 0:
                        b[i] += column[i]
            print("\rsetting a,b: {}%".format( round((ii*m+jj+1)/(m*m), 4)*100 ), end="")
    print("\n")

    electrode_index = np.array([], dtype=np.int)
    for ii in range(m):
        for jj in range(m):
            if boundary[ii,jj] != 100:
                electrode_index = np.append(electrode_index, int(ii*m+jj))
                print("\rdeletion: {}%".format( round((ii*m+jj+1)/(m*m), 4)*100 ), end="")

    a = np.delete(a, electrode_index, 0)
    a = np.delete(a, electrode_index, 1)
    b = np.delete(b, electrode_index, 0)
    row = np.delete(row, electrode_index, 0)
    col = np.delete(col, electrode_index, 0)


    np.savetxt("a.txt", a, fmt='%i', delimiter=" ")
    np.savetxt("b.txt", b, fmt='%i', delimiter=" ")
    np.savetxt("row.txt", row, fmt='%i', delimiter=" ")
    np.savetxt("col.txt", col, fmt='%i', delimiter=" ")

    print("Set ab finished!!")

    return a, b, row, col

def set_index(a0, b0, row, col):
    print("Setting index")
    b_1    = []
    b_10   = []
    b_100  = []
    b_1000 = []
    b_101  = []
    b_1001 = []
    b_110  = []
    b_1010 = []
    a_2i, a_2j       = [], []
    a_3i, a_3j       = [], []
    a_4i, a_4j       = [], []
    a_5i, a_5j       = [], []
    a_6i, a_6j       = [], []
    a_7i, a_7j       = [], []
    a_8i, a_8j       = [], []
    a_9i, a_9j       = [], []
    a_11i, a_11j     = [], []
    a_1i, a_1j       = [], []
    a_10i, a_10j     = [], []
    a_100i, a_100j   = [], []
    a_1000i, a_1000j = [], []

    for i in range(len(a0)):
        if b0[i] == 1: #右
            b_1.append(i)
        elif b0[i] == 10: #左
            b_10.append(i)
        elif b0[i] == 100: #上
            b_100.append(i)
        elif b0[i] == 1000: #下
            b_1000.append(i)
        elif b0[i] == 101: #右上
            b_101.append(i)
        elif b0[i] == 1001: #右下
            b_1001.append(i)
        elif b0[i] == 110: #左上
            b_110.append(i)
        elif b0[i] == 1010: #左下
            b_1010.append(i)

        for j in range(len(a0)):
            if a0[i,j] == 2:
                a_2i.append(i)
                a_2j.append(j)
            elif a0[i,j] == 3:
                a_3i.append(i)
                a_3j.append(j)
            elif a0[i,j] == 4:
                a_4i.append(i)
                a_4j.append(j)
            elif a0[i,j] == 5:
                a_5i.append(i)
                a_5j.append(j)
            elif a0[i,j] == 6:
                a_6i.append(i)
                a_6j.append(j)
            elif a0[i,j] == 7:
                a_7i.append(i)
                a_7j.append(j)
            elif a0[i,j] == 8:
                a_8i.append(i)
                a_8j.append(j)
            elif a0[i,j] == 9:
                a_9i.append(i)
                a_9j.append(j)
            elif a0[i,j] == 11:
                a_11i.append(i)
                a_11j.append(j)
            elif a0[i,j] == 1:
                a_1i.append(i)
                a_1j.append(j)
            elif a0[i,j] == 10:
                a_10i.append(i)
                a_10j.append(j)
            elif a0[i,j] == 100:
                a_100i.append(i)
                a_100j.append(j)
            elif a0[i,j] == 1000:
                a_1000i.append(i)
                a_1000j.append(j)

            print("\rsetting index: {}%".format( round((i*len(a0)+j+1)/(len(a0)*len(a0)), 4)*100 ), end="")

    if b_1 != []:
        np.savetxt("b_1.txt", b_1, fmt='%i', delimiter=" ")
    if b_10 != []:
        np.savetxt("b_10.txt", b_10, fmt='%i', delimiter=" ")
    if b_100 != []:
        np.savetxt("b_100.txt", b_100, fmt='%i', delimiter=" ")
    if b_1000 != []:
        np.savetxt("b_1000.txt", b_1000, fmt='%i', delimiter=" ")
    if b_101 != []:
        np.savetxt("b_101.txt", b_101, fmt='%i', delimiter=" ")
    if b_1001 != []:
        np.savetxt("b_1001.txt", b_1001, fmt='%i', delimiter=" ")
    if b_110 != []:
        np.savetxt("b_110.txt", b_110, fmt='%i', delimiter=" ")
    if b_1010 != []:
        np.savetxt("b_1010.txt", b_1010, fmt='%i', delimiter=" ")
    if a_2i != [] and a_2j != []:
        np.savetxt("a_2i.txt", a_2i, fmt='%i', delimiter=" ")
        np.savetxt("a_2j.txt", a_2j, fmt='%i', delimiter=" ")
    if a_3i != [] and a_3j != []:
        np.savetxt("a_3i.txt", a_3i, fmt='%i', delimiter=" ")
        np.savetxt("a_3j.txt", a_3j, fmt='%i', delimiter=" ")
    if a_4i != [] and a_4j != []:
        np.savetxt("a_4i.txt", a_4i, fmt='%i', delimiter=" ")
        np.savetxt("a_4j.txt", a_4j, fmt='%i', delimiter=" ")
    if a_5i != [] and a_5j != []:
        np.savetxt("a_5i.txt", a_5i, fmt='%i', delimiter=" ")
        np.savetxt("a_5j.txt", a_5j, fmt='%i', delimiter=" ")
    if a_6i != [] and a_6j != []:
        np.savetxt("a_6i.txt", a_6i, fmt='%i', delimiter=" ")
        np.savetxt("a_6j.txt", a_6j, fmt='%i', delimiter=" ")
    if a_7i != [] and a_7j != []:
        np.savetxt("a_7i.txt", a_7i, fmt='%i', delimiter=" ")
        np.savetxt("a_7j.txt", a_7j, fmt='%i', delimiter=" ")
    if a_8i != [] and a_8j != []:
        np.savetxt("a_8i.txt", a_8i, fmt='%i', delimiter=" ")
        np.savetxt("a_8j.txt", a_8j, fmt='%i', delimiter=" ")
    if a_9i != [] and a_9j != []:
        np.savetxt("a_9i.txt", a_9i, fmt='%i', delimiter=" ")
        np.savetxt("a_9j.txt", a_9j, fmt='%i', delimiter=" ")
    if a_11i != [] and a_11j != []:
        np.savetxt("a_11i.txt", a_11i, fmt='%i', delimiter=" ")
        np.savetxt("a_11j.txt", a_11j, fmt='%i', delimiter=" ")
    if a_1i != [] and a_1j != []:
        np.savetxt("a_1i.txt", a_1i, fmt='%i', delimiter=" ")
        np.savetxt("a_1j.txt", a_1j, fmt='%i', delimiter=" ")
    if a_10i != [] and a_10j != []:
        np.savetxt("a_10i.txt", a_10i, fmt='%i', delimiter=" ")
        np.savetxt("a_10j.txt", a_10j, fmt='%i', delimiter=" ")
    if a_100i != [] and a_100j != []:
        np.savetxt("a_100i.txt", a_100i, fmt='%i', delimiter=" ")
        np.savetxt("a_100j.txt", a_100j, fmt='%i', delimiter=" ")
    if a_1000i != [] and a_1000j != []:
        np.savetxt("a_1000i.txt", a_1000i, fmt='%i', delimiter=" ")
        np.savetxt("a_1000j.txt", a_1000j, fmt='%i', delimiter=" ")

    name_a = ["a_2", "a_3", "a_4", "a_5", "a_6", "a_7", "a_8", "a_9", "a_11",
              "a_1", "a_10", "a_100", "a_1000"]
    name_b = ["b_1", "b_10", "b_100", "b_1000", "b_101", "b_1001", "b_110", "b_1010"]

    dict_a = {}
    dict_b = {}
    for name in name_a:
        file_name = name + "i"
        dict_a[file_name] = globals()[file_name]
    
    for name in name_a:
        file_name = name + "j"
        dict_a[file_name] = globals()[file_name]
    
    for name in name_b:
        file_name = name
        dict_b[file_name] = globals()[file_name]

    print("Set index finished!!")
    return dict_a, dict_b

def load_ab(path):
    os.chdir(path)
    if os.path.exists("a.txt"):
        a = np.loadtxt("a.txt", dtype=int)
    else:
        a = []  # np.arrayのindexを空のnp.arrayで指定するとエラーになる

    if os.path.exists("b.txt"):
        b = np.loadtxt("b.txt", dtype=int)
    else:
        b = []

    if os.path.exists("row.txt"):
        row = np.loadtxt("row.txt", dtype=int)
    else:
        row = []

    if os.path.exists("col.txt"):
        col = np.loadtxt("col.txt", dtype=int)
    else:
        col = []

    print("loaded ab.")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # このファイルがあるpathに戻る

    return a, b, row, col

def load_index(path):
    os.chdir(path)
    name_a = ["a_2", "a_3", "a_4", "a_5", "a_6", "a_7", "a_8", "a_9", "a_11",
              "a_1", "a_10", "a_100", "a_1000"]
    name_b = ["b_1", "b_10", "b_100", "b_1000", "b_101", "b_1001", "b_110", "b_1010"]

    for name in name_a:
        file_name = name + "i"
        if os.path.exists(file_name + ".txt"):
            globals()[file_name] = np.loadtxt(file_name + ".txt", dtype=int)  # str -> object　変換
        else:
            globals()[file_name] = []
    
    for name in name_a:
        file_name = name + "j"
        if os.path.exists(file_name + ".txt"):
            globals()[file_name] = np.loadtxt(file_name + ".txt", dtype=int)  # str -> object　変換
        else:
            globals()[file_name] = []
    
    for name in name_b:
        file_name = name
        if os.path.exists(file_name + ".txt"):
            globals()[file_name] = np.loadtxt(file_name + ".txt", dtype=int)  # str -> object　変換
        else:
            globals()[file_name] = []

    dict_a = {}
    dict_b = {}
    for name in name_a:
        file_name = name + "i"
        dict_a[file_name] = globals()[file_name]
    
    for name in name_a:
        file_name = name + "j"
        dict_a[file_name] = globals()[file_name]
    
    for name in name_b:
        file_name = name
        dict_b[file_name] = globals()[file_name]

    print("loaded index.")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # このファイルがあるpathに戻る

    return dict_a, dict_b