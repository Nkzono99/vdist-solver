from collections import deque
from typing import Deque, List, Tuple
from vdsolver.sims.essimulator import ChargedParticle, ESSimulator3d
import emout

import numpy as np
import emout
from .utils import create_simulator


class Target:
    def __init__(self, data: emout.Emout, sim: ESSimulator3d):
        self.data = data
        self.sim = sim

    def solve(self):
        raise NotImplementedError()

    @classmethod
    def load(cls, data: emout.Emout, sim: ESSimulator3d, targetdict: dict):
        return cls(data, sim, **targetdict)


class VSolveTarget(Target):
    def __init__(self,
                 data: emout.Emout,
                 sim: ESSimulator3d,
                 position: List[float],
                 min_velocity: List[float],
                 max_velocity: List[float],
                 nvelocities: List[int],
                 maxstep: int,
                 max_workers: int,
                 chunksize: int,
                 ispec: int,
                 dt: float,
                 istep: int,
                 show_progress: bool = True,
                 ):
        super().__init__(data, sim)
        self.position = position
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.nvelocities = nvelocities
        self.maxstep = maxstep
        self.max_workers = max_workers
        self.chunksize = chunksize
        self.ispec = ispec
        self.dt = dt
        self.istep = istep

        self.show_progress = show_progress
        self.probs = None

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.array(self.position)

        vmin = self.min_velocity
        vmax = self.max_velocity
        nv = self.nvelocities
        vx = np.linspace(vmin[0], vmax[0], nv[0])
        vy = np.linspace(vmin[1], vmax[1], nv[1])
        vz = np.linspace(vmin[2], vmax[2], nv[2])
        VX, VY, VZ = np.meshgrid(vx, vy, vz, indexing='ij')

        vels = np.zeros((len(vz), len(vy), len(vx), 3))
        vels[:, :, :, 0] = VX
        vels[:, :, :, 1] = VY
        vels[:, :, :, 2] = VZ

        dt = self.data.inp.dt * self.dt
        q_m = self.data.inp.qm[self.ispec]
        probs = self.sim.get_probs(pos, vels, q_m, dt,
                                   max_step=self.maxstep,
                                   show_progress=self.show_progress,
                                   use_concurrent=self.max_workers != 1,
                                   max_workers=self.max_workers,
                                   chunksize=self.chunksize)
        self.probs = probs

        return vels, probs


class BackTraceTraget(Target):
    def __init__(self,
                 data,
                 sim,
                 istep: int,
                 ispec: int,
                 position: List[float],
                 velocity: List[float],
                 maxstep: int,
                 dt: int):
        super().__init__(data, sim)
        self.istep = istep
        self.ispec = ispec
        self.position = position
        self.velocity = velocity
        self.maxstep = maxstep
        self.dt = dt

    def solve(self) -> Tuple[Deque[ChargedParticle], float, ChargedParticle]:
        sim = create_simulator(self.data, self.ispec, self.istep)

        pos = np.array(self.position)
        vel = np.array(self.velocity)

        history = deque()

        dt = self.data.inp.dt * self.dt
        q_m = self.data.inp.qm[self.ispec]
        pcl = ChargedParticle(pos, vel, q_m)
        prob, pcl_last = sim.get_prob(
            pcl, dt, max_step=self.maxstep, history=history)

        return history, prob, pcl_last
