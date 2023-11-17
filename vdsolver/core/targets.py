from collections import deque
from typing import Deque, List, Tuple, Union
from vdsolver.core.base import Simulator

import numpy as np
from vdsolver.core import Particle
from vdsolver.sims.essimulator import ChargedParticle, ESSimulator3d
from dataclasses import dataclass


@dataclass
class Lim:
    start: float
    end: float
    num: int

    @classmethod
    def create(cls, val: float):
        return Lim(val, val, 1)
    
    @property
    def delta(self) -> float:
        return (self.end - self.start) / float(self.num)

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

    @property
    def dx(self) -> float:
        return self.xlim.delta

    @property
    def dy(self) -> float:
        return self.ylim.delta

    @property
    def dz(self) -> float:
        return self.zlim.delta

    @property
    def dvx(self) -> float:
        return self.vxlim.delta

    @property
    def dvy(self) -> float:
        return self.vylim.delta

    @property
    def dvz(self) -> float:
        return self.vzlim.delta

    def create_grid(self) -> np.ndarray:
        """Create phase mesh grid.

        Returns
        -------
        np.ndarray
            Phase mesh grid

        Notes
        -----
        Return grid[0:nz, 0:ny, 0:nx, 0:nvz, 0:nvy, 0:nvx, <axis>].

        <axis> = 0: x, 1: y, 2: z, 3: vx, 4: vy, 5: vz
        """
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
class Target:
    sim: Simulator

    def solve(self):
        raise NotImplementedError()


@dataclass
class VSolveTarget(Target):
    sim: Simulator
    pcl_prototype: Particle
    dt: float
    phase_grid: PhaseGrid
    maxstep: int
    max_workers: int
    chunksize: int
    show_progress: bool = True
    use_mpi: bool = False
    adaptive_dt: bool = False

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        phases = self.phase_grid.create_grid()

        pcls = []
        for phase in phases.reshape(-1, phases.shape[-1]):
            pos = phase[:3].copy()
            vel = phase[3:].copy()
            pcl = self.pcl_prototype.craete_clone(pos, vel)
            pcls.append(pcl)

        probs = self.sim.get_probs(pcls=pcls,
                                   dt=self.dt,
                                   max_step=self.maxstep,
                                   max_workers=self.max_workers,
                                   chunksize=self.chunksize,
                                   show_progress=self.show_progress,
                                   use_mpi=self.use_mpi,
                                   adaptive_dt=self.adaptive_dt)
        probs = probs.reshape(phases.shape[:-1])

        return phases, probs


@dataclass
class BackTraceTarget(Target):
    sim: Simulator
    pcl_prototype: Particle
    dt: float
    position: List[float]
    velocity: List[float]
    maxstep: int
    adaptive_dt: bool = False

    def solve(self) -> Tuple[Deque[ChargedParticle], float, ChargedParticle]:
        pos = np.array(self.position)
        vel = np.array(self.velocity)

        history = deque()

        pcl = self.pcl_prototype.craete_clone(pos, vel)
        prob, pcl_last = self.sim.get_prob(
            pcl, self.dt, max_step=self.maxstep, history=history, 
            adaptive_dt=self.adaptive_dt)

        return history, prob, pcl_last
