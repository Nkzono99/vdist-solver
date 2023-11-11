from concurrent import futures
from copy import deepcopy
from typing import Any, Callable, List, Tuple, Union

import numpy as np
from tqdm import tqdm


class Particle:
    """The Particle in the phase space.
    """

    def __init__(self, pos: np.ndarray, vel: np.ndarray, t: float = 0, periodic: bool = False):
        """Initialize the Particle in the phase space.

        Parameters
        ----------
        pos : np.ndarray
            position
        vel : np.ndarray
            velocity
        t : float, optional
            time, by default 0
        periodic : bool, optional
            True when crossing the periodic boundary by default False

            This argment is used to visualize particle orbits (e.g. vdsolver.tools.plot.plot_periodic).
        """
        self.pos = pos
        self.vel = vel
        self.t = t
        self.periodic = periodic

    def __str__(self):
        return 'Particle(p={pos}, v={vel}, t={t})'.format(
            pos=self.pos,
            vel=self.vel,
            t=self.t,
        )

    @classmethod
    def create_prototype(cls, *args, **kwargs):
        pos = np.zeros(3)
        vel = np.zeros(3)
        return cls(pos, vel, *args, **kwargs)

    def craete_clone(self, pos: np.ndarray, vel: np.ndarray):
        return Particle(pos, vel)


class CollisionRecord:
    """Store collision information.
    """

    def __init__(self,
                 boundary: 'Boundary' = None,
                 t: float = 1e10,
                 pcl: Particle = None):
        """Store collision information.

        Parameters
        ----------
        boundary : Boundary, optional
            collided boundary object, by default None
        t : float, optional
            time of collision, by default 1e10
        pcl : Particle, optional
            collided particle, by default None
        """
        self.boundary = boundary
        self.t = t
        self.pcl = pcl

    def update(self, record: 'CollisionRecord'):
        """Update collision information if new info is faster in time.

        Parameters
        ----------
        record : CollisionRecord
            new collision information
        """
        if record is None:
            return
        if self.t < record.t:
            return
        elif self.t == record.t \
                and self.boundary.priority < record.boundary.priority:
            self.boundary = record.boundary
            return
        else:
            self.boundary = record.boundary
            self.t = record.t
            self.pcl = record.pcl


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


class Boundary:
    def __init__(self,
                 prob_func: Prob,
                 priority: int = -1):
        self.prob_func = prob_func
        self._priority = priority

    @property
    def priority(self):
        return self._priority

    def detect_collision(self, pcl: Particle, pcl_next: Particle) -> CollisionRecord:
        """Detect particle-boundary collision.

        Parameters
        ----------
        pcl : Particle
            particle before update
        pcl_next : Particle
            particle after update

        Returns
        -------
        CollisionRecord
            the record including the collision informations
        """
        raise NotImplementedError()

    def get_prob(self, vel: np.ndarray) -> float:
        """Return the probability of existence at the velocity.

        Parameters
        ----------
        vel : np.ndarray
            velocity

        Returns
        -------
        float
            the probability of existence at the velocity
        """
        return self.prob_func(vel)


class BoundaryList(Boundary):
    def __init__(self,
                 boundaries: List[Boundary],
                 priority: int = -1):
        super().__init__(None, priority)
        self.boundaries = boundaries

    def detect_collision(self,
                         pcl: Particle,
                         pcl_next: Particle) -> CollisionRecord:
        record = CollisionRecord()
        for boundary in self.boundaries:
            record_new = boundary.detect_collision(pcl, pcl_next)
            record.update(record_new)
        return None if record.boundary is None else record

    def get_prob(self, vel: np.ndarray) -> float:
        raise Exception('BoundaryList.get_prob is not exists.')

    def expand(self):
        """Expand nested boundary lists to make it faster.
        """
        boundaries_new = []
        for boundary in self.boundaries:
            if isinstance(boundary, BoundaryList):
                boundaries_new += boundary.expand()
            else:
                boundaries_new.append(boundary)
        self.boundaries = boundaries_new
        return self.boundaries


class Field:
    def __call__(self, pos: np.ndarray) -> Any:
        """Return some value at the position.

        Parameters
        ----------
        pos : np.ndarray
            position

        Returns
        -------
        Any
            some value at the position
        """
        raise NotImplementedError()


class FieldScalar(Field):
    def __init__(self,
                 data3d: np.ndarray,
                 dx: float,
                 offsets: np.ndarray = None):
        self.data3d = data3d
        self.dx = dx
        self.offsets = offsets if offsets is not None else np.zeros(3)
        self.offsets = np.array(offsets, dtype=np.float64)
        self.nz, self.ny, self.nx = data3d.shape

    def __call__(self, pos: np.ndarray) -> float:
        """Return scalar value at the position.

        Parameters
        ----------
        pos : np.ndarray
            position

        Returns
        -------
        Any
            scalar value at the position
        """
        lpos = (pos - self.offsets) / self.dx
        ipos = lpos.astype(int)
        rpos = lpos - ipos

        ix, iy, iz = ipos
        ix, iy, iz = ix % self.nx, iy % self.ny, iz % self.nz
        ix1, iy1, iz1 = \
            (ix + 1) % self.nx, (iy + 1) % self.ny, (iz + 1) % self.nz

        rx, ry, rz = rpos
        rx1, ry1, rz1 = 1.0 - rx, 1.0 - ry, 1.0 - rz

        # Linear Interporation
        u00 = rx * self.data3d[iz, iy, ix1] + rx1 * self.data3d[iz, iy, ix]
        u01 = rx * self.data3d[iz, iy1, ix1] + rx1 * self.data3d[iz, iy1, ix]
        u10 = rx * self.data3d[iz1, iy, ix1] + rx1 * self.data3d[iz1, iy, ix]
        u11 = rx * self.data3d[iz1, iy1, ix1] + rx1 * self.data3d[iz1, iy1, ix]

        u0 = ry * u01 + ry1 * u00
        u1 = ry * u11 + ry1 * u10

        u = rz * u1 + rz1 * u0
        return u


class FieldVector3d(Field):
    def __call__(self, pos: np.ndarray) -> np.ndarray:
        """Return vector value at the position.

        Parameters
        ----------
        pos : np.ndarray
            position

        Returns
        -------
        np.ndarray
            vector value at the position
        """
        raise NotImplementedError()


class SimpleFieldVector3d(FieldVector3d):
    def __init__(self, xfield: FieldScalar, yfield: FieldScalar, zfield: FieldScalar):
        """Initialize the simple implemented FieldVector3d object.

        Parameters
        ----------
        xfield : FieldScalar
            x-axis field
        yfield : FieldScalar
            y-axis field
        zfield : FieldScalar
            z-axis field
        """
        self.xfield = xfield
        self.yfield = yfield
        self.zfield = zfield

    def __call__(self, pos: np.ndarray) -> np.ndarray:
        ux = self.xfield(pos)
        uy = self.yfield(pos)
        uz = self.zfield(pos)
        return np.array((ux, uy, uz))


class Simulator:
    def __init__(self, boundary_list: BoundaryList):
        self.boundary_list = boundary_list

    def _backward(self, pcl: Particle, dt: float) -> Particle:
        """Back trace particle.

        Parameters
        ----------
        pcl : Particle
            particle
        dt : float
            simulation time width

        Returns
        -------
        Particle
            updated particle

            this updated particle and parameters' particle have to be different because apply-boundary after this method needs previous and next particles.
        """
        raise NotImplementedError()

    def _apply_boundary(self, pcl: Particle) -> Particle:
        """Apply boundary conditions. 

        This method called after self._backward.

        Parameters
        ----------
        pcl : Particle
            updated particle

        Returns
        -------
        Particle
            boundary-applied particle
        """
        pass

    def get_prob(self,
                 pcl: Particle,
                 dt: float,
                 max_step: int,
                 history: List[Particle] = None,
                 adaptive_dt: bool = False) -> Tuple[float, Particle]:
        """Caluculate and return the probability of existence of the particle.

        Parameters
        ----------
        pcl : Particle
            particle
        dt : float
            simulation time width
        max_step : int
            max steps of simulation
        history : List[Particle], optional
            history list, store pcl-orbit if history is not None, by default None
        adaptive_dt: bool
            True if use adaptive dt (adaptive dt := dt / norm(pcl.vel))

        Returns
        -------
        Tuple[float, Particle]
            the probability of existence and the particle at the last step
        """
        self._apply_boundary(pcl)
        if history is not None:
            history.append(pcl)

        for _ in range(max_step):
            if adaptive_dt:
                tmp_dt = dt / np.linalg.norm(pcl.vel, ord=2)
            else:
                tmp_dt = dt
            pcl_next = self._backward(pcl, tmp_dt)

            record = self.boundary_list.detect_collision(pcl, pcl_next)

            pcl = pcl_next
            self._apply_boundary(pcl)

            if history is not None:
                history.append(pcl)

            if record is not None and record.boundary is not None:
                return record.boundary.get_prob(record.pcl.vel), pcl
        return 0.0, pcl

    def get_probs(self,
                  pcls: List[Particle],
                  dt: float,
                  max_step: int,
                  max_workers: int = 1,
                  chunksize: int = 100,
                  show_progress: bool = True,
                  use_mpi: bool = False,
                  adaptive_dt: bool = False
                  ) -> np.ndarray:
        """Return the probabilities of the existence of the particles.

        Parameters
        ----------
        pcls : List[Particle]
            particles list
        dt : float
            simulation time width
        max_step : int
            max steps of simulation
        max_workers : int, optional
            calculate using process parallelism if max_workers > 1, by default 1
        chunksize : int, optional
            chunksize, this is used if max_workers > 1, by default 100
        show_progress : bool, optional
            show progress if True, by default True

        Returns
        -------
        np.ndarray
            the probabilities of the existence of the particles
        """
        if max_workers <= 1:
            return self._get_probs_serial(
                pcls=pcls,
                dt=dt,
                max_step=max_step,
                show_progress=show_progress,
                adaptive_dt=adaptive_dt)

        if use_mpi:
            return self._get_probs_mpi(
                pcls=pcls,
                dt=dt,
                max_step=max_step,
                max_workers=max_workers,
                chunksize=chunksize,
                show_progress=show_progress,
                adaptive_dt=adaptive_dt
            )
        else:
            return self._get_probs_concurrent(
                pcls=pcls,
                dt=dt,
                max_step=max_step,
                max_workers=max_workers,
                chunksize=chunksize,
                show_progress=show_progress,
                adaptive_dt=adaptive_dt
            )

    def _get_probs_serial(self,
                          pcls: List[Particle],
                          dt: float,
                          max_step: int,
                          show_progress: bool = False,
                          adaptive_dt: bool = False
                          ) -> np.ndarray:
        probs = np.zeros(len(pcls))

        if show_progress:
            pcls = tqdm(pcls)

        for i, pcl in enumerate(pcls):
            prob, _ = self.get_prob(pcl, dt, max_step, adaptive_dt=adaptive_dt)
            probs[i] = prob

        return np.array(probs)

    def _get_probs_concurrent(self,
                              pcls: List[Particle],
                              dt: float,
                              max_step: int,
                              max_workers: int = 4,
                              chunksize: int = 100,
                              show_progress: bool = False,
                              adaptive_dt: bool = False,
                              ) -> np.ndarray:
        probs = np.zeros(len(pcls))

        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            worker = ConcurrentWorker(
                self, dt, max_step, adaptive_dt=adaptive_dt)
            mapped = executor.map(worker,
                                  zip(range(len(pcls)), pcls),
                                  chunksize=chunksize)

            if show_progress:
                mapped = tqdm(mapped, total=len(pcls))

            try:
                for i, prob in mapped:
                    probs[i] = prob
            except KeyboardInterrupt:
                executor.shutdown()
                exit(1)

        return probs

    def _get_probs_mpi(self,
                       pcls: List[Particle],
                       dt: float,
                       max_step: int,
                       max_workers: int = 4,
                       chunksize: int = 100,
                       show_progress: bool = False,
                       adaptive_dt: bool = False,
                       ) -> np.ndarray:
        from mpi4py.futures import MPIPoolExecutor
        probs = np.zeros(len(pcls))

        with MPIPoolExecutor(max_workers=max_workers) as executor:
            worker = ConcurrentWorker(
                self, dt, max_step, adaptive_dt=adaptive_dt)
            mapped = executor.map(worker,
                                  zip(range(len(pcls)), pcls),
                                  chunksize=chunksize)

            if show_progress:
                mapped = tqdm(mapped, total=len(pcls))

            try:
                for i, prob in mapped:
                    probs[i] = prob
            except KeyboardInterrupt:
                executor.shutdown()
                exit(1)

        return probs


class ConcurrentWorker:
    def __init__(self,
                 sim: Simulator,
                 dt: float,
                 max_step: int,
                 adaptive_dt: bool = False):
        self.sim = sim
        self.dt = dt
        self.max_step = max_step
        self.adaptive_dt = adaptive_dt

    def __call__(self, arg: Tuple[int, Particle]) -> Tuple[int, float]:
        """Returns the probability of existence of a particle.

        Parameters
        ----------
        arg : Tuple[int, Particle]
            the index and the particle

        Returns
        -------
        Tuple[int, float]
            the index and probability of existence of the particle
        """
        i, pcl = arg
        prob, _ = self.sim.get_prob(pcl, self.dt, self.max_step,
                                    adaptive_dt=self.adaptive_dt)
        return i, prob
