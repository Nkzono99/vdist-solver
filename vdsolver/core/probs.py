from copy import deepcopy

import numpy as np
from scipy.stats import norm


class Prob:
    def __init__(self, coef) -> None:
        self.coef = coef

    def __call__(self, vel: np.ndarray):
        raise NotImplementedError()

    def __mul__(self, other):
        if isinstance(other, Prob):
            return MulProb(self.copy(), other.copy())
        else:  # other is int or float
            prob_copied = self.copy()
            prob_copied.coef *= other
            return prob_copied

    def __div__(self, other):
        if isinstance(other, Prob):
            raise TypeError()
        else:  # other is int or float
            prob_copied = self.copy()
            prob_copied.coef /= other
            return prob_copied

    def copy(self):
        return deepcopy(self)


class MulProb(Prob):
    def __init__(self, *probs, coef=1.0) -> None:
        super().__init__(coef)
        self.probs = probs

    def __call__(self, vel: np.ndarray):
        p = 1.0
        for prob in self.probs:
            p *= prob(vel)
        return p * self.coef


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
