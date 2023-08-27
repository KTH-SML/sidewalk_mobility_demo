import numpy as np
from typing_extensions import Self
from .spect import Spect

from odp.solver import HJSolver

class LevelSet(Spect):

    # assumes being checked:
    #   - approximation rules
    #   - matching grids

    value_function: np.ndarray

    def __init__(self, vf, grid, exclude=None, *, approximation='exact'):
        super().__init__(approximation)

        self.value_function = vf
        self.grid = grid
        self.exclude = exclude

    def empty(self):
        return np.all(0 < self.value_function)

    def complement(self):
        if self.approximation == 'under':
            approx = 'over'
        elif self.approximation == 'over':
            approx = 'under'
        else:
            approx = 'exact'
        return LevelSet(-self.value_function, 
                        grid = self.grid, 
                        exclude = self.exclude,
                        approximation = approx)

    def union(self, other: Self):
        if self.approximation != 'exact':
            approx = self.approximation
        elif other.approximation != 'exact':
            approx = other.approximation
        else:
            approx = 'exact'
        return LevelSet(np.minimum(self.value_function, other.value_function),
                        grid = self.grid,
                        exclude = self.exclude,
                        approximation = approx)

    def intersect(self, other: Self):
        if self.approximation != 'exact':
            approx = self.approximation
        elif other.approximation != 'exact':
            approx = other.approximation
        else:
            approx = 'exact'
        return LevelSet(np.maximum(self.value_function, other.value_function),
                        grid = self.grid,
                        exclude = self.exclude,
                        approximation = approx)

    def brs(self, timeline, dynamics, save_intermediary=False):
        # target_and_obs = [self.value_function, self.exclude]
        reach_result = HJSolver(dynamics, self.grid, self.value_function, # target_and_obs,
                                timeline.array, {"TargetSetMode": "minVWithV0"},
                                saveAllTimeSteps=save_intermediary)
        return LevelSet(reach_result, approximation = 'under',
                        grid = self.grid, exclude = self.exclude)

    def rci(self, timeline, dynamics, save_intermediary=False):
        # target_and_obs = [-self.value_function, self.exclude]
        avoid_result = HJSolver(dynamics, self.grid, self.value_function, # target_and_obs,
                                timeline.array, {"TargetSetMode": "minVWithV0"},
                                saveAllTimeSteps=save_intermediary)
        return LevelSet(-avoid_result, approximation = 'under',
                        grid = self.grid, exclude = self.exclude)

    def frs(self): pass

    def membership(self, point): pass

    def control_set(self, point): pass


