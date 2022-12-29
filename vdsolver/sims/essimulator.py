import numpy as np

from vdsolver.core import (BoundaryList, FieldVector3d,
                           Particle, Simulator)


class ChargedParticle(Particle):
    def __init__(self,
                 pos: np.ndarray,
                 vel: np.ndarray,
                 q_m: float, t: float = 0,
                 periodic: bool = False):
        super().__init__(pos, vel, t=t, periodic=periodic)
        self.q_m = q_m

    def craete_clone(self, pos: np.ndarray, vel: np.ndarray):
        return ChargedParticle(pos, vel, self.q_m)


class ESSimulator3d(Simulator):
    def __init__(self,
                 nx: int,
                 ny: int,
                 nz: int,
                 dx: float,
                 ef: FieldVector3d,
                 bf: FieldVector3d,
                 boundary_list: BoundaryList):
        super().__init__(boundary_list)
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.ef = ef
        self.bf = bf

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

        if self.bf:
        # Update velocity by Buneman-Boris.
            mdt2 = -0.5*dt
            bf = self.bf(pcl.pos)
            ef = self.ef(pcl.pos)

            t = bf*pcl.q_m*mdt2
            s = 2*t/(1 + t**2)

            # Accerarate by e-field
            upm = pcl.vel + pcl.q_m*ef*mdt2

            # Rotate by b-field
            upa = upm + np.cross(upm, t)
            upp = upm + np.cross(upa, s)

            # Accerarate by e-field
            vel_new = upp + pcl.q_m*ef*mdt2
        else:
            vel_new = pcl.vel - dt * pcl.q_m * self.ef(pcl.pos)

        # Create new particle.
        t_new = pcl.t + dt
        pcl_new = ChargedParticle(pos_new, vel_new, q_m=pcl.q_m, t=t_new)

        return pcl_new
