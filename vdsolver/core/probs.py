import numpy as np
from scipy.stats import norm


class Prob:
    def __call__(self, vel: np.ndarray):
        raise NotImplementedError()


class MaxwellProb:
    def __init__(self, locs, scales):
        self.locs = locs
        self.scales = scales

    def __call__(self, vel):
        p = 1.0
        for i in range(len(self.locs)):
            _p = norm.pdf(vel[i], loc=self.locs[i], scale=self.scales[i])
            p *= _p
        return p


class NoProb:
    def __init__(self):
        pass

    def __call__(self, vel):
        return 0.0
