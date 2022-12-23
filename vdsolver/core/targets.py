from collections import deque
from typing import Deque, List, Tuple, Union
from vdsolver.core.base import Simulator
from vdsolver.core import probs

import emout
import numpy as np
from numpy.lib.arraysetops import isin
from vdsolver.sims.essimulator import ChargedParticle, ESSimulator3d
from dataclasses import dataclass


@dataclass
class Target:
    data: emout.Emout
    sim: ESSimulator3d

    def solve(self):
        raise NotImplementedError()

    @classmethod
    def load(cls, data: emout.Emout, sim: ESSimulator3d, targetdict: dict):
        return cls(data, sim, **targetdict)


@dataclass
class Lim:
    start: float
    end: float
    num: int

    @classmethod
    def create(cls, val: float):
        return Lim(val, val, 1)

    def tolist(self):
        return [self.start, self.end, self.num]


lim_like = Union[Lim, Tuple[float, float, int], float]


@dataclass
class PhaseGrid:
    x: lim_like
    y: lim_like
    z: lim_like
    vx: lim_like
    vy: lim_like
    vz: lim_like

    def _lim(self, val: lim_like) -> Lim:
        if isinstance(val, Lim):
            return val
        if isinstance(val, Tuple):
            return Lim(*val)
        else:
            return Lim.create(val)

    @property
    def xlim(self) -> Lim:
        return self._lim(self.x)

    @property
    def ylim(self) -> Lim:
        return self._lim(self.y)

    @property
    def zlim(self) -> Lim:
        return self._lim(self.z)

    @property
    def vxlim(self) -> Lim:
        return self._lim(self.vx)

    @property
    def vylim(self) -> Lim:
        return self._lim(self.vy)

    @property
    def vzlim(self) -> Lim:
        return self._lim(self.vz)

    def create_grid(self) -> np.ndarray:
        x = np.linspace(*self.xlim.tolist())
        y = np.linspace(*self.ylim.tolist())
        z = np.linspace(*self.zlim.tolist())
        vx = np.linspace(*self.vxlim.tolist())
        vy = np.linspace(*self.vylim.tolist())
        vz = np.linspace(*self.vzlim.tolist())

        Z, Y, X, VZ, VY, VX = np.meshgrid(z, y, x, vz, vy, vx, indexing='ij')

        grd = np.zeros((len(z), len(y), len(x), len(vz), len(vy), len(vx), 6))
        grd[:, :, :, :, :, :, 0] = X
        grd[:, :, :, :, :, :, 1] = Y
        grd[:, :, :, :, :, :, 2] = Z
        grd[:, :, :, :, :, :, 3] = VX
        grd[:, :, :, :, :, :, 4] = VY
        grd[:, :, :, :, :, :, 5] = VZ

        return grd


@dataclass
class VSolveTarget(Target):
    data: emout.Emout
    sim: ESSimulator3d
    phase_grid: PhaseGrid
    maxstep: int
    max_workers: int
    chunksize: int
    ispec: int
    dt: float
    istep: int
    show_progress: bool = True
    use_mpi: bool = False

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        phases = self.phase_grid.create_grid()

        dt = self.data.inp.dt * self.dt
        q_m = self.data.inp.qm[self.ispec]

        pcls = []
        for phase in phases.reshape(-1, phases.shape[-1]):
            pos = phase[:3]
            vel = phase[3:]
            pcl = ChargedParticle(pos, vel, q_m)
            pcls.append(pcl)

        probs = self.sim.get_probs(pcls=pcls,
                                   dt=dt,
                                   max_step=self.maxstep,
                                   max_workers=self.max_workers,
                                   chunksize=self.chunksize,
                                   show_progress=self.show_progress,
                                   use_mpi=self.use_mpi)
        probs = probs.reshape(phases.shape[:-1])

        return phases, probs


@dataclass
class BackTraceTraget(Target):
    data: emout.Emout
    sim: ESSimulator3d
    istep: int
    ispec: int
    position: List[float]
    velocity: List[float]
    maxstep: int
    dt: int

    def solve(self) -> Tuple[Deque[ChargedParticle], float, ChargedParticle]:
        pos = np.array(self.position)
        vel = np.array(self.velocity)

        history = deque()

        dt = self.data.inp.dt * self.dt
        q_m = self.data.inp.qm[self.ispec]
        pcl = ChargedParticle(pos, vel, q_m)
        prob, pcl_last = self.sim.get_prob(
            pcl, dt, max_step=self.maxstep, history=history)

        return history, prob, pcl_last
