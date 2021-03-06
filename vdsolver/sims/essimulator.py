from concurrent import futures

import numpy as np
from tqdm import tqdm
from vdsolver.core import *


class ChargedParticle(Particle):
    def __init__(self,
                 pos: np.ndarray,
                 vel: np.ndarray,
                 q_m: float, t: float = 0,
                 periodic: bool = False):
        super().__init__(pos, vel, t=t, periodic=periodic)
        self.q_m = q_m


class ESSimulator3d(Simulator):
    def __init__(self,
                 nx: int,
                 ny: int,
                 nz: int,
                 dx: float,
                 ef: FieldVector3d,
                 boundary_list: BoundaryList):
        super().__init__(boundary_list)
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.ef = ef

    def _apply_boundary(self, pcl: Particle) -> Particle:
        px, py, pz = pcl.pos
        pcl.pos[0] = np.mod(pcl.pos[0], self.nx*self.dx)
        pcl.pos[1] = np.mod(pcl.pos[1], self.ny*self.dx)
        pcl.pos[2] = np.mod(pcl.pos[2], self.nz*self.dx)
        pcl.periodic = (pcl.pos[0] != px) \
            or (pcl.pos[1] != py) \
            or (pcl.pos[2] != pz)

    def _backward(self, pcl: ChargedParticle, dt: float) -> ChargedParticle:
        pos_new = pcl.pos - dt * pcl.vel
        vel_new = pcl.vel - dt * pcl.q_m * self.ef(pcl.pos)
        t_new = pcl.t + dt
        pcl_new = ChargedParticle(pos_new, vel_new, q_m=pcl.q_m, t=t_new)
        return pcl_new
