import numpy as np


class fdtd_2d_TMz_systolic_block:
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
        self.sigma = 0
        self.khx1 = 0
        self.kex1 = 0
        self.khx2 = 0
        self.kex2 = 0
        self.khy1 = 0
        self.key1 = 0
        self.khy2 = 0
        self.key2 = 0

    def set_block(self, dt, dx, dy, ep, mu, sigma=0):
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.ep = ep
        self.mu = mu
        self.sigma = sigma
        self.kex1 = (ep - 0.5 * sigma * dt) / (ep + 0.5 * sigma * dt)
        self.khx1 = (mu - 0.5 * sigma * mu / ep * dt) / (mu + 0.5 * sigma * mu / ep * dt)
        self.kex2 = dt / dy / (ep + 0.5 * sigma * dt)
        self.khx2 = dt / dy / (mu + 0.5 * sigma * mu / ep * dt)
        self.key1 = (ep - 0.5 * sigma * dt) / (ep + 0.5 * sigma * dt)
        self.khy1 = (mu - 0.5 * sigma * mu / ep * dt) / (mu + 0.5 * sigma * mu / ep * dt)
        self.key2 = dt / dx / (ep + 0.5 * sigma * dt)
        self.khy2 = dt / dx / (mu + 0.5 * sigma * mu / ep * dt)

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
        self.Ezx = self.kex1 * self.Ezx + self.khx2 * (self.Hy - Hy)

    def update_Ezy(self, Hx):
        self.Ezy = self.key1 * self.Ezy + self.key2 * (self.Hx - Hx)

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
        # self.systolic_blocks2 = []
        for i in range(x_nodes):
            row1 = []
            # col2 = []
            for j in range(y_nodes):
                row1.append(fdtd_2d_TMz_systolic_block())
                row1[j].set_block(dt, dx, dy, ep=ep, mu=mu, sigma=0)
                # col2.append(fdtd_2d_TMz_systolic_block())
                # col2[i].set_block(dt, dx, dy, ep=ep, mu=mu)

                # self.systolic_blocks2.append(col2)
            self.systolic_blocks1.append(row1.copy())
            del row1
        self.Ez = np.zeros([x_nodes, y_nodes])
        self.Ezx = np.zeros([x_nodes, y_nodes])
        self.Ezy = np.zeros([x_nodes, y_nodes])
        self.Hx = np.zeros([x_nodes, y_nodes])
        self.Hy = np.zeros([x_nodes, y_nodes])

    def apply_src(self, pos, value, stype='E'):
        x = pos[0]
        y = pos[1]
        self.systolic_blocks1[x][y].apply_src(value, stype)

    def update(self):
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                if j == self.y_nodes - 1:
                    self.systolic_blocks1[i][j].update_Hx(0)
                else:
                    self.systolic_blocks1[i][j].update_Hx(self.systolic_blocks1[i][j + 1].Ez)

        for j in range(self.y_nodes):
            for i in range(self.x_nodes):
                if i == self.x_nodes - 1:
                    self.systolic_blocks1[i][j].update_Hy(0)
                else:
                    self.systolic_blocks1[i][j].update_Hy(self.systolic_blocks1[i + 1][j].Ez)
        for j in range(self.y_nodes):
            for i in range(self.x_nodes):
                if i == 0:
                    self.systolic_blocks1[i][j].update_Ezx(0)
                else:
                    self.systolic_blocks1[i][j].update_Ezx(self.systolic_blocks1[i - 1][j].Hy)

        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                if j == 0:
                    self.systolic_blocks1[i][j].update_Ezy(0)
                else:
                    self.systolic_blocks1[i][j].update_Ezy(self.systolic_blocks1[i][j - 1].Hx)

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
                    self.systolic_blocks1[i][j].set_block(self.dt, self.dx, sigma=Pright[d - 1 - i], mu=self.mu, ep=self.ep)
                    #self.systolic_blocks2[i][j].set_block(self.dt, self.dx, sigma=Pright[d - 1 - i], mu=self.mu, ep=self.ep)
        else:
            for i in range(d):
                self.systolic_blocks1[self.nodes - d + i].set_block(self.dt, self.dx, sigma=Pright[i], mu=self.mu, ep=self.ep)
                #self.systolic_blocks2[self.nodes - d + i].set_block(self.dt, self.dx, sigma=Pright[i], mu=self.mu, ep=self.ep)

    def export_value(self):
        for i in range(self.x_nodes):
            for j in range(self.y_nodes):
                self.Ez[i][j] = self.systolic_blocks1[i][j].Ez
        return self.Ez


'''
    def add_material(self,ep,mu,sigma,place):
        ss = place[0]
        ee = place[1]
        for i in range(ss,ee):
            self.systolic_blocks2[i].set_block(self.dt,self.dx,ep=ep,mu=mu,sigma=sigma)

    def set_tfsf_boundary(self,place):
        self.tfsf_left = place[0]
        self.tfsf_right = place[1]
'''
