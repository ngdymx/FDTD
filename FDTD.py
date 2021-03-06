import numpy as np


class fdtd_1d_systolic_block:
    H = None
    E = None

    def __init__(self):
        self.E = 0
        self.H = 0
        self.dt = 0
        self.dx = 0
        self.ep = 0
        self.mu = 0
        self.sigma = 0
        self.kh1 = 0
        self.ke1 = 0
        self.kh2 = 0
        self.ke2 = 0

    def set_block(self, dt, dx, ep, mu, sigma=0):
        self.dt = dt
        self.dx = dx
        self.ep = ep
        self.mu = mu
        self.sigma = sigma
        self.ke2 = dt / dx / (ep + 0.5 * sigma * dt)
        self.kh2 = dt / dx / (mu + 0.5 * sigma * mu / ep * dt)
        self.ke1 = (ep - 0.5 * sigma * dt) / (ep + 0.5 * sigma * dt)
        self.kh1 = (mu - 0.5 * sigma * mu / ep * dt) / (mu + 0.5 * sigma * mu / ep * dt)

    def update_parameters(self):
        self.ke2 = self.dt / self.dx / (self.ep + 0.5 * self.sigma * self.dt)
        self.kh2 = self.dt / self.dx / (self.mu + 0.5 * self.sigma * self.mu / self.ep * self.dt)
        self.ke1 = (self.ep - 0.5 * self.sigma * self.dt) / (self.ep + 0.5 * self.sigma * self.dt)
        self.kh1 = (self.mu - 0.5 * self.sigma * self.mu / self.ep * self.dt) / (self.mu + 0.5 * self.sigma * self.mu / self.ep * self.dt)

    def apply_src(self, value, stype='E'):
        if stype == 'E':
            self.E = self.E + value
        else:
            self.H = self.H + value

    def update_E(self, H):
        self.E = self.ke1 * self.E - self.ke2 * (self.H - H)

    def update_H(self, E):
        self.H = self.kh1 * self.H - self.kh2 * (E - self.E)


class fdtd_2d_TEz_systolic_block:
    Hz = None

    def __init__(self):
        self.Hz = 0
        self.Hzx = 0
        self.Hzy = 0
        self.Ex = 0
        self.Ey = 0
        self.dt = 0
        self.dx = 0
        self.dy = 0
        self.ep = 0
        self.mu = 0
        self.sigma_x = 0
        self.sigma_y = 0
        self.khx1 = 0
        self.kex1 = 0
        self.khx2 = 0
        self.kex2 = 0
        self.khy1 = 0
        self.key1 = 0
        self.khy2 = 0
        self.key2 = 0

    def set_block(self, dt, dx, dy, ep, mu, sigma_x=0, sigma_y=0):
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.ep = ep
        self.mu = mu
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.kex1 = (ep - 0.5 * sigma_x * dt) / (ep + 0.5 * sigma_x * dt)
        self.kex2 = dt / dy / (ep + 0.5 * sigma_x * dt)
        self.key1 = (ep - 0.5 * sigma_y * dt) / (ep + 0.5 * sigma_y * dt)
        self.key2 = dt / dx / (ep + 0.5 * sigma_y * dt)
        self.khx1 = (mu - 0.5 * sigma_x * mu / ep * dt) / (mu + 0.5 * sigma_x * mu / ep * dt)
        self.khx2 = dt / dx / (mu + 0.5 * sigma_x * mu / ep * dt)
        self.khy1 = (mu - 0.5 * sigma_y * mu / ep * dt) / (mu + 0.5 * sigma_y * mu / ep * dt)
        self.khy2 = dt / dy / (mu + 0.5 * sigma_y * mu / ep * dt)

    def update_parameters(self):
        self.kex1 = (self.ep - 0.5 * self.sigma_x * self.dt) / (self.ep + 0.5 * self.sigma_x * self.dt)
        self.kex2 = self.dt / self.dy / (self.ep + 0.5 * self.sigma_x * self.dt)
        self.key1 = (self.ep - 0.5 * self.sigma_y * self.dt) / (self.ep + 0.5 * self.sigma_y * self.dt)
        self.key2 = self.dt / self.dx / (self.ep + 0.5 * self.sigma_y * self.dt)
        self.khx1 = (self.mu - 0.5 * self.sigma_x * self.mu / self.ep * self.dt) / \
                    (self.mu + 0.5 * self.sigma_x * self.mu / self.ep * self.dt)
        self.khx2 = self.dt / self.dx / (self.mu + 0.5 * self.sigma_x * self.mu / self.ep * self.dt)
        self.khy1 = (self.mu - 0.5 * self.sigma_y * self.mu / self.ep * self.dt) / \
                    (self.mu + 0.5 * self.sigma_y * self.mu / self.ep * self.dt)
        self.khy2 = self.dt / self.dy / (self.mu + 0.5 * self.sigma_y * self.mu / self.ep * self.dt)

    def apply_src(self, value, stype='H'):
        if stype == 'H':
            self.Hz = self.Hz + value
            self.Hzx = self.Hzx + value / 2
            self.Hzy = self.Hzy + value / 2
        else:
            self.Ex = self.Ex + value
            self.Ey = self.Ey + value

    def update_Ex(self, Hz):
        self.Ex = self.kex1 * self.Ex + self.kex2 * (self.Hz - Hz)

    def update_Ey(self, Hz):
        self.Ey = self.key1 * self.Ey - self.key2 * (self.Hz - Hz)

    def update_Hzx(self, Ey):
        self.Hzx = self.khx1 * self.Hzx - self.khx2 * (Ey - self.Ey)

    def update_Hzy(self, Ex):
        self.Hzy = self.khy1 * self.Hzy + self.khy2 * (Ex - self.Ex)

    def update_Hz(self):
        self.Hz = self.Hzx + self.Hzy


class FDTD_2D_TEz_space:
    def __init__(self, x_nodes, y_nodes, dt, dx, dy, mu, ep):
        self.x_nodes = x_nodes
        self.y_nodes = y_nodes
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.mu = mu
        self.ep = ep
        self.systolic_blocks1 = []
        self.systolic_blocks2 = []
        for i in range(x_nodes):
            col1 = []
            for j in range(y_nodes):
                col1.append(fdtd_2d_TEz_systolic_block())
                col1[j].set_block(dt, dx, dy, ep=ep, mu=mu, sigma_x=0, sigma_y=0)
            self.systolic_blocks1.append(col1.copy())
            self.systolic_blocks2.append(fdtd_1d_systolic_block())
            self.systolic_blocks2[i].set_block(self.dt, self.dx, ep=ep, mu=mu, sigma=0)
            del col1

        self.E = np.zeros([x_nodes, 1])
        self.H = np.zeros([x_nodes, 1])

        self.Hz = np.zeros([x_nodes, y_nodes])
        self.Hzx = np.zeros([x_nodes, y_nodes])
        self.Hzy = np.zeros([x_nodes, y_nodes])
        self.Ex = np.zeros([x_nodes, y_nodes])
        self.Ey = np.zeros([x_nodes, y_nodes])

    def apply_src(self, pos, value, stype='H'):
        self.systolic_blocks2[pos].apply_src(value, stype)

    def update(self):
        # update 2D Ex
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                if j == 0:
                    self.systolic_blocks1[i][j].update_Ex(0)
                else:
                    self.systolic_blocks1[i][j].update_Ex(self.systolic_blocks1[i][j - 1].Hz)

        # update 2D Ey
        for j in range(self.y_nodes):
            for i in range(self.x_nodes):
                if i == 0:
                    self.systolic_blocks1[i][j].update_Ey(0)
                else:
                    self.systolic_blocks1[i][j].update_Ey(self.systolic_blocks1[i - 1][j].Hz)

        # correct 2D Ex
        for i in range(self.tfsf_top, self.tfsf_bottom):
            self.systolic_blocks1[i][self.tfsf_left].Ex = self.systolic_blocks1[i][self.tfsf_left].Ex - \
                                                          self.dt / self.dy / self.ep * self.systolic_blocks2[i].H
            self.systolic_blocks1[i][self.tfsf_right].Ex = self.systolic_blocks1[i][self.tfsf_right].Ex + \
                                                           self.dt / self.dy / self.ep * self.systolic_blocks2[i].H

        # correct 2D Ey
        for j in range(self.tfsf_left, self.tfsf_right):
            self.systolic_blocks1[self.tfsf_top][j].Ey = self.systolic_blocks1[self.tfsf_top][j].Ey + self.systolic_blocks2[self.tfsf_top - 1].ke2 * self.systolic_blocks2[self.tfsf_top - 1].H
            self.systolic_blocks1[self.tfsf_bottom][j].Ey = self.systolic_blocks1[self.tfsf_bottom][j].Ey - \
                                                            self.dt / self.ep / self.dx * self.systolic_blocks2[self.tfsf_bottom].H

        # update 1D E
        for i in range(self.x_nodes):
            if i == 0:
                self.systolic_blocks2[i].update_E(0)
            else:
                self.systolic_blocks2[i].update_E(self.systolic_blocks2[i - 1].H)

        # update 1D H
        for i in range(self.x_nodes):
            if i == self.x_nodes - 1:
                self.systolic_blocks2[i].update_H(0)
            else:
                self.systolic_blocks2[i].update_H(self.systolic_blocks2[i + 1].E)

        # correct 2D Hzx, Hzy
        for j in range(self.tfsf_left, self.tfsf_right):
            self.systolic_blocks1[self.tfsf_top - 1][j].Hzx = self.systolic_blocks1[self.tfsf_top - 1][j].Hzx + self.systolic_blocks2[self.tfsf_top].kh2 * self.systolic_blocks2[self.tfsf_top].E / 2
            self.systolic_blocks1[self.tfsf_bottom][j].Hzx = self.systolic_blocks1[self.tfsf_bottom][j].Hzx - \
                                                             self.dt / self.dx / self.mu * self.systolic_blocks2[
                                                                 self.tfsf_bottom].E / 2
            self.systolic_blocks1[self.tfsf_top - 1][j].Hzy = self.systolic_blocks1[self.tfsf_top - 1][j].Hzy + self.systolic_blocks2[self.tfsf_top].kh2 * self.systolic_blocks2[self.tfsf_top].E / 2
            self.systolic_blocks1[self.tfsf_bottom][j].Hzy = self.systolic_blocks1[self.tfsf_bottom][j].Hzy - \
                                                             self.dt / self.dx / self.mu * self.systolic_blocks2[
                                                                 self.tfsf_bottom].E / 2
        # update 2D Hzx
        for j in range(self.y_nodes):
            for i in range(self.x_nodes):
                if i == self.x_nodes - 1:
                    self.systolic_blocks1[i][j].update_Hzx(0)
                else:
                    self.systolic_blocks1[i][j].update_Hzx(self.systolic_blocks1[i + 1][j].Ey)

        # update 2D Hzy
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                if j == self.y_nodes - 1:
                    self.systolic_blocks1[i][j].update_Hzy(0)
                else:
                    self.systolic_blocks1[i][j].update_Hzy(self.systolic_blocks1[i][j + 1].Ex)

        # update 2D Hz
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                self.systolic_blocks1[i][j].update_Hz()

    def export_value_Hz(self):
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                self.Hz[i][j] = self.systolic_blocks1[i][j].Hz
        return self.Hz

    def export_value_H(self):
        for i in range(self.x_nodes):
            self.H[i] = self.systolic_blocks2[i].H
        return self.H

    def set_tfsf_boundary(self, first_node, last_node):
        self.tfsf_top = first_node[0]
        self.tfsf_left = first_node[1]
        self.tfsf_bottom = last_node[0]
        self.tfsf_right = last_node[1]

    def set_pml(self, y_side, x_side, d, R0=1e-16, M=3, ep=8.85e-12):
        sigmax_max = -np.log10(R0) * (M + 1) * ep * 3e8 / 2 / d / self.dx
        sigmay_max = -np.log10(R0) * (M + 1) * ep * 3e8 / 2 / d / self.dy
        Pright = np.power((np.arange(d) / d), M) * sigmax_max
        Ptop = np.power((np.arange(d) / d), M) * sigmay_max
        if x_side == 'L':
            for i in range(d):
                for j in range(self.y_nodes):
                    self.systolic_blocks1[i][j].sigma_x = Pright[d - 1 - i]
                    self.systolic_blocks2[i].sigma = Pright[d - 1 - i]
                    self.systolic_blocks2[i].update_parameters()
        else:
            for i in range(d):
                for j in range(self.y_nodes):
                    self.systolic_blocks1[self.x_nodes - d + i][j].sigma_x = Pright[i]
                    self.systolic_blocks2[self.x_nodes - d + i].sigma = Pright[i]
                    self.systolic_blocks2[i].update_parameters()
        if y_side == 'T':
            for j in range(d):
                for i in range(self.x_nodes):
                    self.systolic_blocks1[i][j].sigma_y = Ptop[d - 1 - j]
        else:
            for j in range(d):
                for i in range(self.x_nodes):
                    self.systolic_blocks1[i][self.y_nodes - d + j].sigma_y = Ptop[j]
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                self.systolic_blocks1[i][j].update_parameters()

    def add_material(self, ep, mu, sigma, R):
        #ss1 = place1[0]
        #ss2 = place1[1]
        #ee1 = place2[0]
        #ee2 = place2[1]
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                if (i - self.x_nodes/2)**2 + (j - self.y_nodes/2)**2 < R**2:
                    self.systolic_blocks1[i][j].set_block(self.dt, self.dx, self.dy, ep=ep, mu=mu, sigma_x=sigma, sigma_y=sigma)


class fdtd_2d_TMz_systolic_block:
    Ez = None

    def __init__(self):
        self.Ez = 0
        self.Ezx = 0
        self.Ezy = 0
        self.Hx = 0
        self.Hy = 0
        self.dt = 0
        self.dx = 0
        self.dy = 0
        self.ep = 0
        self.mu = 0
        self.sigma_x = 0
        self.sigma_y = 0
        self.khx1 = 0
        self.kex1 = 0
        self.khx2 = 0
        self.kex2 = 0
        self.khy1 = 0
        self.key1 = 0
        self.khy2 = 0
        self.key2 = 0

    def set_block(self, dt, dx, dy, ep, mu, sigma_x=0, sigma_y=0):
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.ep = ep
        self.mu = mu
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.kex1 = (ep - 0.5 * sigma_x * dt) / (ep + 0.5 * sigma_x * dt)
        self.khx1 = (mu - 0.5 * sigma_x * mu / ep * dt) / (mu + 0.5 * sigma_x * mu / ep * dt)
        self.kex2 = dt / dy / (ep + 0.5 * sigma_x * dt)
        self.khx2 = dt / dy / (mu + 0.5 * sigma_x * mu / ep * dt)
        self.key1 = (ep - 0.5 * sigma_y * dt) / (ep + 0.5 * sigma_y * dt)
        self.khy1 = (mu - 0.5 * sigma_y * mu / ep * dt) / (mu + 0.5 * sigma_y * mu / ep * dt)
        self.key2 = dt / dx / (ep + 0.5 * sigma_y * dt)
        self.khy2 = dt / dx / (mu + 0.5 * sigma_y * mu / ep * dt)

    def update_parameters(self):
        self.kex1 = (self.ep - 0.5 * self.sigma_x * self.dt) / (self.ep + 0.5 * self.sigma_x * self.dt)
        self.khx1 = (self.mu - 0.5 * self.sigma_x * self.mu / self.ep * self.dt) / (
                self.mu + 0.5 * self.sigma_x * self.mu / self.ep * self.dt)
        self.kex2 = self.dt / self.dy / (self.ep + 0.5 * self.sigma_x * self.dt)
        self.khx2 = self.dt / self.dy / (self.mu + 0.5 * self.sigma_x * self.mu / self.ep * self.dt)
        self.key1 = (self.ep - 0.5 * self.sigma_y * self.dt) / (self.ep + 0.5 * self.sigma_y * self.dt)
        self.khy1 = (self.mu - 0.5 * self.sigma_y * self.mu / self.ep * self.dt) / (
                self.mu + 0.5 * self.sigma_y * self.mu / self.ep * self.dt)
        self.key2 = self.dt / self.dx / (self.ep + 0.5 * self.sigma_y * self.dt)
        self.khy2 = self.dt / self.dx / (self.mu + 0.5 * self.sigma_y * self.mu / self.ep * self.dt)

    def apply_src(self, value, stype='E'):
        if stype == 'E':
            self.Ez = self.Ez + value
            self.Ezx = self.Ezx + value / 2
            self.Ezy = self.Ezy + value / 2
        else:
            self.Hx = self.Hx + value
            self.Hy = self.Hy + value

    def update_Hx(self, Ez):
        self.Hx = self.khx1 * self.Hx - self.khx2 * (Ez - self.Ez)

    def update_Hy(self, Ez):
        self.Hy = self.khy1 * self.Hy + self.khy2 * (Ez - self.Ez)

    def update_Ezx(self, Hy):
        self.Ezx = self.kex1 * self.Ezx + self.kex2 * (self.Hy - Hy)

    def update_Ezy(self, Hx):
        self.Ezy = self.key1 * self.Ezy - self.key2 * (self.Hx - Hx)

    def update_Ez(self):
        self.Ez = self.Ezx + self.Ezy


class FDTD_2D_TMz_space:
    def __init__(self, x_nodes, y_nodes, dt, dx, dy, mu, ep):
        self.x_nodes = x_nodes
        self.y_nodes = y_nodes
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.mu = mu
        self.ep = ep
        self.systolic_blocks1 = []
        self.systolic_blocks2 = []
        for i in range(x_nodes):
            col1 = []
            for j in range(y_nodes):
                col1.append(fdtd_2d_TMz_systolic_block())
                col1[j].set_block(dt, dx, dy, ep=ep, mu=mu, sigma_x=0, sigma_y=0)
            self.systolic_blocks1.append(col1.copy())
            self.systolic_blocks2.append(fdtd_1d_systolic_block())
            self.systolic_blocks2[i].set_block(self.dt, self.dx, ep=ep, mu=mu, sigma=0)
            del col1

        self.E = np.zeros([x_nodes, 1])
        self.H = np.zeros([x_nodes, 1])

        self.Ez = np.zeros([x_nodes, y_nodes])
        self.Ezx = np.zeros([x_nodes, y_nodes])
        self.Ezy = np.zeros([x_nodes, y_nodes])
        self.Hx = np.zeros([x_nodes, y_nodes])
        self.Hy = np.zeros([x_nodes, y_nodes])

    def apply_src(self, pos, value, stype='E'):
        self.systolic_blocks2[pos].apply_src(value, stype)

    def update(self):
        # update 2D Hx
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                if j == self.y_nodes - 1:
                    self.systolic_blocks1[i][j].update_Hx(0)
                else:
                    self.systolic_blocks1[i][j].update_Hx(self.systolic_blocks1[i][j + 1].Ez)

        # update 2D Hy
        for j in range(self.y_nodes):
            for i in range(self.x_nodes):
                if i == self.x_nodes - 1:
                    self.systolic_blocks1[i][j].update_Hy(0)
                else:
                    self.systolic_blocks1[i][j].update_Hy(self.systolic_blocks1[i + 1][j].Ez)

        # correct 2D Hx
        for i in range(self.tfsf_top, self.tfsf_bottom):
            self.systolic_blocks1[i][self.tfsf_left - 1].Hx = self.systolic_blocks1[i][self.tfsf_left - 1].Hx + \
                                                              self.dt / self.dy / self.mu * self.systolic_blocks2[i].E
            self.systolic_blocks1[i][self.tfsf_right - 1].Hx = self.systolic_blocks1[i][self.tfsf_right - 1].Hx - \
                                                               self.dt / self.dy / self.mu * self.systolic_blocks2[i].E
        # correct 2D Hy
        for j in range(self.tfsf_left, self.tfsf_right):
            self.systolic_blocks1[self.tfsf_top - 1][j].Hy = self.systolic_blocks1[self.tfsf_top - 1][j].Hy - \
                                                             self.dt / self.mu / self.dx * self.systolic_blocks2[
                                                                 self.tfsf_top].E
            self.systolic_blocks1[self.tfsf_bottom - 1][j].Hy = self.systolic_blocks1[self.tfsf_bottom - 1][j].Hy + \
                                                                self.dt / self.mu / self.dx * self.systolic_blocks2[
                                                                    self.tfsf_bottom - 1].E

        # update 1D H
        for i in range(self.x_nodes):
            if i == self.x_nodes - 1:
                self.systolic_blocks2[i].update_H(0)
            else:
                self.systolic_blocks2[i].update_H(self.systolic_blocks2[i + 1].E)
        # update 1D E
        for i in range(self.x_nodes):
            if i == 0:
                self.systolic_blocks2[i].update_E(0)
            else:
                self.systolic_blocks2[i].update_E(self.systolic_blocks2[i - 1].H)

        # correct 2D Ezx, Ezy
        for j in range(self.tfsf_left, self.tfsf_right):
            self.systolic_blocks1[self.tfsf_top][j].Ezx = self.systolic_blocks1[self.tfsf_top][j].Ezx - \
                                                          self.dt / self.dx / self.ep * self.systolic_blocks2[
                                                              self.tfsf_top - 1].H / 2
            self.systolic_blocks1[self.tfsf_bottom - 1][j].Ezx = self.systolic_blocks1[self.tfsf_bottom - 1][j].Ezx + \
                                                                 self.dt / self.dx / self.ep * self.systolic_blocks2[
                                                                     self.tfsf_bottom - 1].H / 2
            # correct 2D Ezy
            self.systolic_blocks1[self.tfsf_top][j].Ezy = self.systolic_blocks1[self.tfsf_top][j].Ezy - \
                                                          self.dt / self.dx / self.ep * self.systolic_blocks2[
                                                              self.tfsf_top - 1].H / 2
            self.systolic_blocks1[self.tfsf_bottom - 1][j].Ezy = self.systolic_blocks1[self.tfsf_bottom - 1][j].Ezy + \
                                                                 self.dt / self.dx / self.ep * self.systolic_blocks2[
                                                                     self.tfsf_bottom - 1].H / 2

        # update 2D Ezx
        for j in range(self.y_nodes):
            for i in range(self.x_nodes):
                if i == 0:
                    self.systolic_blocks1[i][j].update_Ezx(0)
                else:
                    self.systolic_blocks1[i][j].update_Ezx(self.systolic_blocks1[i - 1][j].Hy)

        # update 2D Ezy
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                if j == 0:
                    self.systolic_blocks1[i][j].update_Ezy(0)
                else:
                    self.systolic_blocks1[i][j].update_Ezy(self.systolic_blocks1[i][j - 1].Hx)

        # update 2D Ez
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                self.systolic_blocks1[i][j].update_Ez()

    def set_pml(self, y_side, x_side, d, R0=1e-16, M=3, ep=8.85e-12):
        sigmax_max = -np.log10(R0) * (M + 1) * ep * 3e8 / 2 / d / self.dx
        sigmay_max = -np.log10(R0) * (M + 1) * ep * 3e8 / 2 / d / self.dy
        Pright = np.power((np.arange(d) / d), M) * sigmax_max
        Ptop = np.power((np.arange(d) / d), M) * sigmay_max
        if x_side == 'L':
            for i in range(d):
                for j in range(self.y_nodes):
                    self.systolic_blocks1[i][j].sigma_x = Pright[d - 1 - i]
                    self.systolic_blocks2[i].sigma = Pright[d - 1 - i]
                    self.systolic_blocks2[i].update_parameters()
        else:
            for i in range(d):
                for j in range(self.y_nodes):
                    self.systolic_blocks1[self.x_nodes - d + i][j].sigma_x = Pright[i]
                    self.systolic_blocks2[self.x_nodes - d + i].sigma = Pright[i]
                    self.systolic_blocks2[i].update_parameters()

        if y_side == 'T':
            for j in range(d):
                for i in range(self.x_nodes):
                    self.systolic_blocks1[i][j].sigma_y = Ptop[d - 1 - j]
        else:
            for j in range(d):
                for i in range(self.x_nodes):
                    self.systolic_blocks1[i][self.y_nodes - d + j].sigma_y = Ptop[j]
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                self.systolic_blocks1[i][j].update_parameters()

    def export_value_Ez(self):
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                self.Ez[i][j] = self.systolic_blocks1[i][j].Ez
        return self.Ez

    def export_value_E(self):
        for i in range(self.x_nodes):
            self.E[i] = self.systolic_blocks2[i].E
        return self.E

    def set_tfsf_boundary(self, first_node, last_node):
        self.tfsf_top = first_node[0]
        self.tfsf_left = first_node[1]
        self.tfsf_bottom = last_node[0]
        self.tfsf_right = last_node[1]

    def add_material(self, ep, mu, sigma, R):
        # ss1 = place1[0]
        # ss2 = place1[1]
        # ee1 = place2[0]
        # ee2 = place2[1]
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                if (i - self.x_nodes / 2) ** 2 + (j - self.y_nodes / 2) ** 2 < R ** 2:
                    self.systolic_blocks1[i][j].set_block(self.dt, self.dx, self.dy, ep=ep, mu=mu, sigma_x=sigma,
                                                          sigma_y=sigma)
