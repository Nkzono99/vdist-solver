from typing import List
from .base import Particle
import matplotlib.pyplot as plt


def plot_periodic(history: List[Particle], idxs=[0, 2]):
    ims = []

    path = []
    n = 0
    while n < len(history):
        if history[n].periodic or len(history) - n == 1:
            x = [pcl.pos[idxs[0]] for pcl in path]
            z = [pcl.pos[idxs[1]] for pcl in path]
            im = plt.plot(x, z, color='steelblue')
            ims.append(im)
            path = [history[n]]
        else:
            path.append(history[n])
        n += 1

    return ims
