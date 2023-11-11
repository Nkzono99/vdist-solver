from scipy.stats import norm
from .base import Prob


class MaxwellProb(Prob):
    def __init__(self, locs, scales, coef=1.0):
        super().__init__(coef)
        self.locs = locs
        self.scales = scales

    def __call__(self, vel):
        p = 1.0
        for i in range(len(self.locs)):
            _p = norm.pdf(vel[i], loc=self.locs[i], scale=self.scales[i])
            p *= _p
        return p * self.coef


class NoProb(Prob):
    def __init__(self):
        super().__init__(0)

    def __call__(self, vel):
        return 0.0
