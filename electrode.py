# 電極形状を定義する関数群
# 引数は必ず(boundary, psi, V, psi0)に統一する
# 実装時には、クラスSystemの__init__()の引数electrodeに渡される

from numba.decorators import jit

@jit
def four_electrode(boundary, psi, V, psi0):
    """
    Define the shape of electrode.
    Here, number of mesh N+1 = len(boundary) should be 170 (or N = 169).
    Ideally, electrode length : memristor length = 50 : 117, but to save memory
    consumption, it is defined as 26 : 59.
    electrode length : memristor length = 52 : 118 = 26 : 59
    length of each mesh is 2 um.
    length (hight) of electrode = 59 mesh = 118 um
    length between electrode = 26 mesh = 52 um
    """
    #electrode 1
    for i in range(13, 73):
        for j in range(13, 73):
            boundary[i,j] = V
            psi[i,j] = V
    #electrode 3
    for i in range(98, 158):
        for j in range(98, 158):
            boundary[i,j] = psi0
            psi[i,j] = psi0
    #electrode 2
    for i in range(13, 73):
        for j in range(98, 158):
            boundary[i,j] = V
            psi[i,j] = V
    #electrode 4
    for i in range(98, 158):
        for j in range(13, 73):
            boundary[i,j] = V
            psi[i,j] = V

    return boundary, psi


@jit
def two_electrode_13(boundary, psi, V, psi0):
    #electrode 1
    for i in range(13, 73):
        for j in range(13, 73):
            boundary[i,j] = V
            psi[i,j] = V
    #electrode 3
    for i in range(98, 158):
        for j in range(98, 158):
            boundary[i,j] = psi0
            psi[i,j] = psi0

    return boundary, psi

@jit
def two_electrode_24(boundary, psi, V, psi0):
    #electrode 2
    for i in range(13, 73):
        for j in range(98, 158):
            boundary[i,j] = V
            psi[i,j] = V
    #electrode 4
    for i in range(98, 158):
        for j in range(13, 73):
            boundary[i,j] = V
            psi[i,j] = V

    return boundary, psi