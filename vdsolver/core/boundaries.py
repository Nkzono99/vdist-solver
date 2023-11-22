from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np

from .base import Boundary, BoundaryList, CollisionRecord, Particle


class Plane2d(Boundary):
    def __init__(self,
                 idx: int,
                 val: float,
                 func_prob: Callable[[np.ndarray], float],
                 priority: int = 0):
        super().__init__(func_prob, priority)
        self.idx = idx
        self.val = val

    def detect_collision(self,
                         pcl: Particle,
                         pcl_next: Particle,
                         ) -> CollisionRecord:
        d1 = pcl.pos[self.idx] - self.val
        d2 = pcl_next.pos[self.idx] - self.val
        if d1 * d2 >= 0:
            return None

        r = abs(d1) / (abs(d1) + abs(d2))
        t = (1 - r) * pcl.t + r * pcl_next.t
        pos = (1 - r) * pcl.pos + r * pcl_next.pos
        vel = (1 - r) * pcl.vel + r * pcl_next.vel
        _pcl = Particle(pos, vel, t)
        return CollisionRecord(self, t, _pcl)


class PlaneXY(Plane2d):
    def __init__(self,
                 z: float,
                 func_prob: Callable[[np.ndarray], float],
                 priority: int = 0):
        super().__init__(2, z, func_prob, priority=priority)


class PlaneYZ(Plane2d):
    def __init__(self,
                 x: float,
                 func_prob: Callable[[np.ndarray], float],
                 priority: int = 0):
        super().__init__(0, x, func_prob, priority=priority)


class PlaneZX(Plane2d):
    def __init__(self,
                 y: float,
                 func_prob: Callable[[np.ndarray], float],
                 priority: int = 0):
        super().__init__(1, y, func_prob, priority=priority)


class ParallelRectangle(Boundary):
    def __init__(self,
                 idxs: Tuple[int, int, int],
                 pos: np.ndarray,
                 w1: float,
                 w2: float,
                 func_prob: Callable[[np.ndarray], float],
                 priority: int = 0,
                 ):
        super().__init__(func_prob, priority)
        self.idxs = idxs
        self.pos = pos
        self.w1 = w1
        self.w2 = w2

    def detect_collision(self,
                         pcl: Particle,
                         pcl_next: Particle,
                         ) -> CollisionRecord:
        d1 = pcl.pos[self.idxs[0]] - self.pos[self.idxs[0]]
        d2 = pcl_next.pos[self.idxs[0]] - self.pos[self.idxs[0]]
        if d1 * d2 >= 0:
            return None

        r = abs(d1) / (abs(d1) + abs(d2))
        t = (1 - r) * pcl.t + r * pcl_next.t
        pos = (1 - r) * pcl.pos + r * pcl_next.pos

        if pos[self.idxs[1]] < self.pos[self.idxs[1]]  \
                or self.pos[self.idxs[1]] + self.w1 < pos[self.idxs[1]]:
            return None

        if pos[self.idxs[2]] < self.pos[self.idxs[2]] \
                or self.pos[self.idxs[2]] + self.w2 < pos[self.idxs[2]]:
            return None

        vel = (1 - r) * pcl.vel + r * pcl_next.vel
        _pcl = Particle(pos, vel, t)
        return CollisionRecord(self, t, _pcl)


class RectangleX(ParallelRectangle):
    def __init__(self,
                 pos: np.ndarray,
                 dy: float,
                 dz: float,
                 func_prob: Callable[[np.ndarray], float],
                 priority: int = 0,
                 ):
        super().__init__((0, 1, 2), pos, dy, dz, func_prob, priority)


class RectangleY(ParallelRectangle):
    def __init__(self,
                 pos: np.ndarray,
                 dz: float,
                 dx: float,
                 func_prob: Callable[[np.ndarray], float],
                 priority: int = 0,):
        super().__init__((1, 2, 0), pos, dz, dx, func_prob, priority)


class RectangleZ(ParallelRectangle):
    def __init__(self,
                 pos: np.ndarray,
                 dx: float,
                 dy: float,
                 func_prob: Callable[[np.ndarray], float],
                 priority: int = 0,):
        super().__init__((2, 0, 1), pos, dx, dy, func_prob, priority)


def create_simbox(xlim: Tuple[float, float],
                  ylim: Tuple[float, float],
                  zlim: Tuple[float, float],
                  func_prob_default: Callable[[np.ndarray], float],
                  func_prob_dict: Dict[str,
                                       Callable[[np.ndarray], float]] = {},
                  priority_prob_dict: Dict[str, int] = {},
                  use_wall: List[str] = None,
                  ) -> BoundaryList:
    if use_wall is None:
        use_wall = ['xl', 'xu', 'yl', 'yu', 'zl', 'zu']

    fpdict = defaultdict(lambda: func_prob_default)
    ppdict = defaultdict(lambda: -1)
    for key, value in func_prob_dict.items():
        fpdict[key] = value
    for key, value in priority_prob_dict.items():
        ppdict[key] = value

    boundaries = []
    if 'xl' in use_wall:
        xl = PlaneYZ(xlim[0], fpdict['xl'], priority=ppdict['xl'])
        boundaries.append(xl)
    if 'xu' in use_wall:
        xu = PlaneYZ(xlim[1], fpdict['xu'], priority=ppdict['xu'])
        boundaries.append(xu)
    if 'yl' in use_wall:
        yl = PlaneZX(ylim[0], fpdict['yl'], priority=ppdict['yl'])
        boundaries.append(yl)
    if 'yu' in use_wall:
        yu = PlaneZX(ylim[1], fpdict['yu'], priority=ppdict['yu'])
        boundaries.append(yu)
    if 'zl' in use_wall:
        zl = PlaneXY(zlim[0], fpdict['zl'], priority=ppdict['zl'])
        boundaries.append(zl)
    if 'zu' in use_wall:
        zu = PlaneXY(zlim[1], fpdict['zu'], priority=ppdict['zu'])
        boundaries.append(zu)

    box = BoundaryList(boundaries)
    return box


def create_box(xlim: Tuple[float, float],
               ylim: Tuple[float, float],
               zlim: Tuple[float, float],
               func_prob_default: Callable[[np.ndarray], float],
               func_prob_dict: Dict[str, Callable[[np.ndarray], float]],
               priority_prob_dict: Dict[str, int],
               use_wall: List[str] = None,
               ) -> BoundaryList:
    if use_wall is None:
        use_wall = ['xl', 'xu', 'yl', 'yu', 'zl', 'zu']

    fpdict = defaultdict(lambda: func_prob_default)
    ppdict = defaultdict(lambda: 1)
    for key, value in func_prob_dict.items():
        fpdict[key] = value
    for key, value in priority_prob_dict.items():
        ppdict[key] = value

    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]
    dz = zlim[1] - zlim[0]

    boundaries = []

    if 'xl' in use_wall:
        org = np.array([xlim[0], ylim[0], zlim[0]])
        xl = RectangleX(org, dx, dy, fpdict['xl'], priority=ppdict['xl'])
        boundaries.append(xl)
    if 'xu' in use_wall:
        org = np.array([xlim[1], ylim[0], zlim[0]])
        xu = RectangleX(org, dx, dy, fpdict['xu'], priority=ppdict['xu'])
        boundaries.append(xu)
    if 'yl' in use_wall:
        org = np.array([xlim[0], ylim[0], zlim[0]])
        yl = RectangleY(org, dz, dx, fpdict['yl'], priority=ppdict['yl'])
        boundaries.append(yl)
    if 'yu' in use_wall:
        org = np.array([xlim[0], ylim[1], zlim[0]])
        yu = RectangleY(org, dz, dx, fpdict['yu'], priority=ppdict['yu'])
        boundaries.append(yu)
    if 'zl' in use_wall:
        org = np.array([xlim[0], ylim[0], zlim[0]])
        zl = RectangleZ(org, dx, dy, fpdict['zl'], priority=ppdict['zl'])
        boundaries.append(zl)
    if 'zu' in use_wall:
        org = np.array([xlim[0], ylim[0], zlim[1]])
        zu = RectangleZ(org, dx, dy, fpdict['zu'], priority=ppdict['zu'])
        boundaries.append(zu)

    box = BoundaryList(boundaries)
    return box
