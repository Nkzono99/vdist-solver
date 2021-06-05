from typing import Any, Callable, List, Tuple

import numpy as np


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
        self.boundary = record.boundary
        self.t = record.t
        self.pcl = record.pcl


class Boundary:
    def __init__(self, func_prob: Callable[[np.ndarray], float]):
        self._func_prob = func_prob

    def detect_collision(self, pcl: Particle, pcl_next: Particle) -> CollisionRecord:
        raise NotImplementedError()

    def get_prob(self, vel: np.ndarray) -> float:
        return self._func_prob(vel)


class BoundaryList(Boundary):
    def __init__(self, boundaries: List[Boundary]):
        self.boundaries = boundaries

    def detect_collision(self, pcl: Particle, pcl_next: Particle) -> CollisionRecord:
        record = CollisionRecord()
        for boundary in self.boundaries:
            record_new = boundary.detect_collision(pcl, pcl_next)
            record.update(record_new)
        return None if record.boundary is None else record

    def get_prob(self, vel: np.ndarray) -> float:
        raise Exception('BoundaryList.get_prob is not exists.')

    def expand(self):
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
        raise NotImplementedError()


class FieldScalar(Field):
    def __init__(self,
                 data3d: np.ndarray,
                 dx: float,
                 offsets: np.ndarray = None):
        self.data3d = data3d
        self.dx = dx
        self.offsets = offsets if offsets is not None else np.zeros(3)
        self.nz, self.ny, self.nx = data3d.shape

    def __call__(self, pos: np.ndarray) -> float:
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
        raise NotImplementedError()


class SimpleFieldVector3d(FieldVector3d):
    def __init__(self, xfield: FieldScalar, yfield: FieldScalar, zfield: FieldScalar):
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
        raise NotImplementedError()

    def _apply_boundary(self, pcl: Particle) -> Particle:
        pass

    def get_probs(self,
                  pos: np.ndarray,
                  vels: np.ndarray,
                  dt: float,
                  max_step: int) -> float:
        probs = []

        for vel in vels.reshape(-1, vels.shape[-1]):
            pcl = Particle(pos, vel)
            prob, _ = self.get_prob(pcl, dt, max_step)
            probs.append(prob)
        return np.array(probs).reshape(vels.shape[:-1])

    def get_prob(self,
                 pcl: Particle,
                 dt: float,
                 max_step: int,
                 history: List[Particle] = None) -> Tuple[float, Particle]:
        self._apply_boundary(pcl)
        if history is not None:
            history.append(pcl)

        for _ in range(max_step):
            pcl_next = self._backward(pcl, dt)

            record = self.boundary_list.detect_collision(pcl, pcl_next)

            pcl = pcl_next
            self._apply_boundary(pcl)

            if history is not None:
                history.append(pcl)

            if record is not None and record.boundary is not None:
                return record.boundary.get_prob(record.pcl.vel), pcl
        return 0.0, pcl
