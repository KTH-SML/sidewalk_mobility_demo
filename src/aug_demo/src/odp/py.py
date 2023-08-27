import numpy as np
import heterocl as hcl
from odp.grid import Grid
from odp.model import Model
from odp.solver import HJSolver
from odp.shapes import rectangle

class Dynamics(Model):

    state_dims = 2
    ctrl_dims = 2
    dstb_dims = state_dims

    X, Y = range(state_dims)
    VX, VY = range(ctrl_dims)

    def opt_ctrl(self, u, dv, t, x):
        with hcl.if_(0 <= dv[self.X]):
            u[self.VX] = -1
        with hcl.else_():
            u[self.VX] = +1

        with hcl.if_(0 <= dv[self.Y]):
            u[self.VY] = -1
        with hcl.else_():
            u[self.VY] = +1

    def opt_dstb(self, d, dv, t, x):
        with hcl.if_(0 <= dv[self.X]):
            d[self.VX] = +1 
        with hcl.else_():
            d[self.VX] = -1

        with hcl.if_(0 <= dv[self.Y]):
            d[self.VY] = +1 
        with hcl.else_():
            d[self.VY] = -1

    def dynamics(self, dx, t, x, u, d):
        dx[self.X] = u[self.VX] + d[self.X]
        dx[self.Y] = u[self.VY] + d[self.Y]

if __name__ == "__main__":
    
    grid = Grid([ 0,  0],
                [10, 10],
                [20, 20])

    model = Dynamics()

    # graph = ComputeGraph(grid, model)
    # s = graph.build()

    tau = np.linspace(0, 10, 10)

    target = rectangle(grid, 
                       [0, 0], 
                       [1, 1])

    solver = HJSolver(grid, tau, model, target=target, target_mode='minVWithVTarget')

