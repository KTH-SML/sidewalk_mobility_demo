import numpy as np
import heterocl as hcl
import odp.shapes
import odp.grid

from dataclasses import dataclass
from odp.model import Model
from pyspect import LevelSet

@dataclass
class Timeline: 
    start: float
    stop: float 
    resolution: float = 0.1

    @property
    def array(self):
        return np.arange(start=self.start,
                         stop=self.stop + 1e-5,
                         step=self.resolution)

class Grid: 

    _names: dict
    _maxs: np.ndarray
    _mins: np.ndarray
    _shape: np.ndarray
    _is_periodic: np.ndarray

    def __init__(self, names, maxs, mins, shape, is_periodic=None):
        self._names = {name: idx for idx, name in enumerate(names)}
        self._shape = np.asarray(shape)
        self._dims = len(self._shape)

        self._maxs = np.asarray(maxs)
        self._mins = np.asarray(mins)

        if is_periodic is None:
            self._is_periodic = np.zeros(self._shape, dtype=bool) 
        else:
            self._is_periodic = np.asarray(is_periodic, dtype=bool)

        assert len(self._maxs) == len(self._shape)
        assert len(self._mins) == len(self._shape)
        assert len(self._is_periodic) == len(self._shape) 

        self._grid = odp.grid.Grid(self._mins, 
                                   self._maxs,
                                   self._shape,
                                   [i for i, p in enumerate(self._is_periodic) if p])

    def __getattr__(self, name: str):
        if name in self._names:
            return self._names[name] 

        if name.startswith('max_'):
            name = name[4:]
            if name in self._names:
                i = self._names[name]
                return self._maxs[i]

        if name.startswith('min_'):
            name = name[4:]
            if name in self._names:
                i = self._names[name]
                return self._mins[i]

        if name.startswith('len_'):
            name = name[4:]
            if name in self._names:
                i = self._names[name]
                return self._shape[i]

        raise AttributeError(f'{type(self).__name__} does not have attribute {name}')

    def axes(self):
        return range(self._dims)

    def cylinder(self, *args, **kwargs):
        return LevelSet(odp.shapes.cylinder(self._grid, *args, **kwargs), self._grid)

    def box(self, axes, maxs, mins, inf=True):
        grid_lims = np.asarray([self._mins, self._maxs])
        box_lims = np.asarray([mins, maxs])
        for i, a in enumerate(axes):
            grid_lims[:, a] = box_lims[:, i]
        area = self.rectangle(grid_lims[0, :], grid_lims[1, :])
        if inf:
            area.value_function[area.value_function < 0] = -np.inf
        return area

    def rectangle(self, *args, **kwargs):
        return LevelSet(odp.shapes.rectangle(self._grid, *args, **kwargs), self._grid)

    def point(self, *args, **kwargs):
        return LevelSet(odp.shapes.point(self._grid, *args, **kwargs), self._grid)

    def above(self, *args, **kwargs): 
        return self.upper_half_space(*args, **kwargs)

    def below(self, *args, **kwargs): 
        return self.lower_half_space(*args, **kwargs)

    def lower_half_space(self, *args, **kwargs):
        return LevelSet(odp.shapes.lower_half_space(self._grid, *args, **kwargs), self._grid)

    def upper_half_space(self, *args, **kwargs):
        return LevelSet(odp.shapes.upper_half_space(self._grid, *args, **kwargs), self._grid)

class SVEA(Model):

    state_dims = 4 
    ctrl_dims = 2 
    dstb_dims = state_dims

    wheelbase = 0.32

    X, Y, YAW, VEL = range(state_dims)
    STEERING, VELOCITY = range(ctrl_dims)

    def __init__(self, ctrl_range, dstb_range, mode='reach') -> None:

        self.ctrl_range = np.asarray(ctrl_range)
        assert self.ctrl_range.shape[1] == self.ctrl_dims

        self.dstb_range = np.asarray(dstb_range)
        assert self.dstb_range.shape[1] == self.dstb_dims

        modes = {'reach': {"uMode": "min", "dMode": "max"},
                 'avoid': {"uMode": "max", "dMode": "min"}}
        self.mode = modes[mode]

    def dynamics(self, dx, t, x, u, d):

        # x_dot = v * cos(theta) + d_1
        dx[self.X] = (x[self.VEL] * hcl.cos(x[self.YAW]) 
                      + d[self.X])

        # y_dot = v * sin(theta) + d_2
        dx[self.Y] = (x[self.VEL] * hcl.sin(x[self.YAW])
                      + d[self.Y])

        # theta_dot = (v * tan(u1))/L + d3
        hcl_tan = lambda a: hcl.sin(a) / hcl.cos(a)
        dx[self.YAW] = (x[self.VEL] * hcl_tan(u[self.STEERING])/self.wheelbase
                        + d[self.YAW])

        # v_dot = u2 + d4
        dx[self.VEL] = u[self.VELOCITY] + d[self.VEL]

    def opt_ctrl(self, u, dv, t, x):

        uMin, uMax = self.ctrl_range
        
        if self.mode['uMode'] == "max":
            # Steering
            with hcl.if_(0 <= x[self.VEL]):
                with hcl.if_(0 <= dv[self.YAW]):
                    u[self.STEERING] = uMax[self.STEERING]
                with hcl.else_():
                    u[self.STEERING] = uMin[self.STEERING]
            with hcl.else_():
                with hcl.if_(0 <= dv[self.YAW]):
                    u[self.STEERING] = uMin[self.STEERING]
                with hcl.else_():
                    u[self.STEERING] = uMax[self.STEERING]
            # Velocity
            with hcl.if_(0 <= dv[self.VEL]):
                u[self.VELOCITY] = uMax[self.VELOCITY]
            with hcl.else_():
                u[self.VELOCITY] = uMin[self.VELOCITY]            
        else:
            # Steering
            with hcl.if_(0 <= x[self.VEL]):
                with hcl.if_(0 <= dv[self.YAW]):
                    u[self.STEERING] = uMin[self.STEERING]
                with hcl.else_():
                    u[self.STEERING] = uMax[self.STEERING]
            with hcl.else_():
                with hcl.if_(0 <= dv[self.YAW]):
                    u[self.STEERING] = uMax[self.STEERING]
                with hcl.else_():
                    u[self.STEERING] = uMin[self.STEERING]
            # Velocity
            with hcl.if_(0 <= dv[self.VEL]):
                u[self.VELOCITY] = uMin[self.VELOCITY]
            with hcl.else_():
                u[self.VELOCITY] = uMax[self.VELOCITY]

    def opt_dstb(self, d, dv, t, x):

        dMin, dMax = self.dstb_range

        for i in range(self.dstb_dims):
            if self.mode['dMode'] == "max":
                with hcl.if_(0 <= dv[i]):
                    d[i] = dMax[i]
                with hcl.else_():
                    d[i] = dMin[i]
            else:
                with hcl.if_(0 <= dv[i]):
                    d[i] = dMin[i]
                with hcl.else_():
                    d[i] = dMax[i]

class ControlledSVEA(SVEA):

    state_dims = 5
    ctrl_dims = 2 
    dstb_dims = state_dims

    wheelbase = 0.32

    X, Y, YAW, VEL, DEL = range(state_dims)
    STEERING, VELOCITY = range(ctrl_dims)

    def dynamics(self, dx, t, x, u, d):

        # x_dot = v * cos(theta) + d_1
        dx[self.X] = (x[self.VEL] * hcl.cos(x[self.YAW]) 
                      + d[self.X])

        # y_dot = v * sin(theta) + d_2
        dx[self.Y] = (x[self.VEL] * hcl.sin(x[self.YAW])
                      + d[self.Y])

        # theta_dot = (v * tan(u1))/L + d3
        hcl_tan = lambda a: hcl.sin(a) / hcl.cos(a)
        dx[self.YAW] = (x[self.VEL] * hcl_tan(u[self.STEERING])/self.wheelbase
                        + d[self.YAW])

        # v_dot = u2 + d4
        dx[self.VEL] = u[self.VELOCITY] + d[self.VEL]

        # delta_dot = u1 + d5
        dx[self.DEL] = u[self.STEERING] + d[self.DEL]
