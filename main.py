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
    space1 = FDTD.FDTD_2D_TEz_space(50, 200, dt, dx, dy, ep=ep, mu=mu)
    space1.set_pml('T', 'L', 5)
    space1.set_pml('B', 'R', 5)
    space1.set_tfsf_boundary([10, 30], [40, 170])
    space1.add_material(ep * 8, mu, 0, 10)

    # space2 = FDTD.FDTD_2D_TMz_space(50, 200, dt, dx, dy, ep=ep, mu=mu)
    # space2.set_pml('T', 'L', 5)
    # space2.set_pml('B', 'R', 5)
    # space2.set_tfsf_boundary([10, 30], [40, 170])
    # space2.add_material(ep * 8, mu, 0, 10)

    figure1 = plt.figure(figsize=(6.4,4.8),dpi=300)
    X = np.arange(400)
    Y = np.arange(400)
    X, Y = np.meshgrid(X, Y)

    for t in range(Steps):
        plt.clf()
        space1.update()
        space1.apply_src(8, 10*np.sin(2 * np.pi * t / 20), stype='H')
        Hz = space1.export_value_Hz()
        plt.imshow(Hz, vmin=-1, vmax=1)
        # space2.update()
        # space2.apply_src(8, 10 * np.sin(2 * np.pi * t / 20), stype='E')
        # Ez = space2.export_value_Ez()
        # plt.imshow(Ez, vmin=-1, vmax=1)
        # plt.plot(Hz)
        # plt.ylim((-10,10))
        plt.pause(.01)
        #print(t)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
