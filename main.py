import FDTD
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dx = 0.1
    dy = 0.1
    Steps = 200
    c = 3e8
    mu = 4 * np.pi * 1e-7
    ep = 1 / mu / c / c
    dt = 0.8 * 1 / np.sqrt(1/(dx**2) + 1/(dy**2)) / 3e8
    space = FDTD.FDTD_2D_TMz_space(100, 100, dt, dx, dy, ep=ep, mu=mu)
    space.set_pml('T', 'L', 20)
    space.set_pml('B', 'R', 20)
    space.set_tfsf_boundary([30, 30], [70, 70])
    '''
    space.set_pml('L', 50)
    space.set_pml('R', 50)
    space.add_material(ep*8,mu,0,[150,200])
    space.set_tfsf_boundary([100,300])
    '''
    figure = plt.figure(dpi=100)
    X = np.arange(400)
    Y = np.arange(400)
    X, Y = np.meshgrid(X, Y)
    #axes = figure.add_subplot(1,1,1)
    #axes = Axes3D(figure)
    for t in range(Steps):
        plt.clf()
        space.update()
        space.apply_src(9, 10*np.sin(2 * np.pi * t / 20))
        Ez = space.export_value()
        plt.imshow(Ez, vmin=-1, vmax=1)
        plt.pause(.0002)
        #print(t)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
