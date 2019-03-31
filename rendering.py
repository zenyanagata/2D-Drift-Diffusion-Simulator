# レンダリング用関数
# 各自出力したい形式にrender関数内を変更する。

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mpl_toolkits.axes_grid1
import matplotlib.animation as animation


import constant as C
import parameter as P


def render(path_to_save, psi, Nd, j, p, n, boundary, count, vapp):

    if not os.path.exists(path_to_save + "/figure"):
                    os.makedirs(path_to_save + "/figure")
    
    if not os.path.exists(path_to_save + "/text"):
                    os.makedirs(path_to_save + "/text")

    padded_count = '{0:05d}'.format(count)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    im = ax.imshow(psi, cmap="rainbow")
    fig.colorbar(im, cax=cax)
    fig.savefig("figure/psi_"+padded_count+
                "_V="+str(vapp)+".png", format="png", dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    im = ax.imshow(Nd/P.h**3/P.Nd0_, cmap="rainbow")
    fig.colorbar(im, cax=cax)
    fig.savefig("figure/Nd_"+padded_count+
                "_V="+str(vapp)+".png", format="png", dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    im = ax.imshow(Nd/P.h**3/P.Nd0_, cmap="rainbow", 
                    norm=LogNorm(vmin=10**-4, vmax=1))
    fig.colorbar(im, cax=cax)
    fig.savefig("figure/Nd_log_"+padded_count+
                "_V="+str(vapp)+".png", format="png", dpi=300)
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    im = ax.imshow(j/P.J0_, cmap="rainbow")
    fig.colorbar(im, cax=cax)
    fig.savefig("figure/J_"+padded_count+"_V="+
                str(vapp)+".png", format="png", dpi=300)
    plt.close()

    np.savetxt('text/psi_'+padded_count+"_V="+str(vapp)+
                '.txt', psi*C.VT, fmt='%15.8e', delimiter=" ")
    np.savetxt('text/p_'+padded_count+"_V="+str(vapp)+
                '.txt', p/P.h**3, fmt='%15.8e', delimiter=" ")
    np.savetxt('text/n_'+padded_count+"_V="+str(vapp)+
                '.txt', n/P.h**3, fmt='%15.8e', delimiter=" ")
    np.savetxt('text/Nd_'+padded_count+"_V="+str(vapp)+
                '.txt', Nd/P.h**3, fmt='%15.8e', delimiter=" ")
    np.savetxt('text/J_'+padded_count+"_V="+str(vapp)+
                '.txt', j/P.J0_, fmt='%15.8e', delimiter=" ")

    Nd_forplot = Nd.copy()
    temp = np.reshape(Nd.copy(), len(Nd)*len(Nd))
    for i in range(0, len(Nd)):
        for j in range(0, len(Nd)):
            if boundary[i, j] != 100:  # if electrode, no change in psi.
                temp[i*len(Nd) + j] = 0
    Nd_ave_ = np.average(temp)

    for i in range(len(Nd_forplot)):
        for j in range(len(Nd_forplot)):
            if boundary[i,j] != 100:
                Nd_forplot[i,j] = Nd_ave_
    plt.figure()
    plt.imshow(Nd_forplot, cmap='rainbow')
    plt.title("Nd")
    plt.savefig("figure/Nd_ave_"+padded_count+"_V="+str(vapp)+".png", format = 'png', dpi=300)
    plt.close()


def from_txt_to_gif(saved_path, vmax, vmin, keyword="*Nd_*.txt"):
    os.chdir(saved_path)
    files = glob.glob(keyword)
    print("{} file found.".format(len(files)))

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    images = []
    for file in files:
        array = np.genfromtxt(file)
        im = ax.imshow(array, cmap="rainbow", vmax=1, vmin=-1)
        fig.colorbar(im, cax=cax)
        images.append(im)
        
    ani = animation.ArtistAnimation(fig, images, interval=100)  # interval -> ms
    plt.show()
    ani.save("output.gif", writer="imagemagick")