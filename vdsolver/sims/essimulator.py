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

    def get_probs(self,
                  pos: np.ndarray,
                  vels: np.ndarray,
                  q_m: float,
                  dt: float,
                  max_step: int,
                  show_progress: bool = False,
                  max_workers: int = 4,
                  chunksize: int = 1,
                  use_concurrent=True) -> float:
        if use_concurrent:
            return self._get_probs_concurrent(
                pos,
                vels,
                q_m,
                dt,
                max_step,
                show_progress=show_progress,
                max_workers=max_workers,
                chunksize=chunksize,
            )
        else:
            return self._get_probs_serial(
                pos,
                vels,
                q_m,
                dt,
                max_step=max_step,
                show_progress=show_progress,
            )

    def _get_probs_concurrent(self,
                              pos: np.ndarray,
                              vels: np.ndarray,
                              q_m: float,
                              dt: float,
                              max_step: int,
                              show_progress: bool = False,
                              max_workers: int = 4,
                              chunksize: int = 1):
        vels_it = vels.reshape(-1, vels.shape[-1])

        probs = np.zeros(len(vels_it))

        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            worker = ESWorker(self, pos, q_m, dt, max_step)
            mapped = executor.map(worker, zip(
                range(len(vels_it)), vels_it), chunksize=chunksize)
            if show_progress:
                mapped = tqdm(mapped, total=len(vels_it))
            try:
                for i, prob in mapped:
                    probs[i] = prob
            except KeyboardInterrupt:
                executor.shutdown()
                exit(1)

        return probs.reshape(vels.shape[:-1])

    def _get_probs_serial(self,
                          pos: np.ndarray,
                          vels: np.ndarray,
                          q_m: float,
                          dt: float,
                          max_step: int,
                          show_progress: bool = False):
        it = vels.reshape(-1, vels.shape[-1])
        if show_progress:
            it = tqdm(it)

        probs = np.zeros(len(it))

        for i, vel in enumerate(it):
            pcl = ChargedParticle(pos, vel, q_m)
            prob, _ = self.get_prob(pcl, dt, max_step)
            probs[i] = prob

        return probs.reshape(vels.shape[:-1])

    def _backward(self, pcl: ChargedParticle, dt: float) -> Particle:
        pos_new = pcl.pos - dt * pcl.vel
        vel_new = pcl.vel - dt * pcl.q_m * self.ef(pcl.pos)
        t_new = pcl.t + dt
        pcl_new = ChargedParticle(pos_new, vel_new, q_m=pcl.q_m, t=t_new)
        return pcl_new


class ESWorker:
    def __init__(self,
                 sim: ESSimulator3d,
                 pos: np.ndarray,
                 q_m: float,
                 dt: float,
                 max_step: int):
        self.sim = sim
        self.pos = pos
        self.q_m = q_m
        self.dt = dt
        self.max_step = max_step

    def __call__(self, arg: Tuple[int, np.ndarray]) -> Tuple[int, float]:
        i, vel = arg
        pcl = ChargedParticle(self.pos, vel, self.q_m)
        prob, _ = self.sim.get_prob(pcl, self.dt, self.max_step)
        return i, prob
