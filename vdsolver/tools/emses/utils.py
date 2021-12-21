
from vdsolver.core.probs import NoProb

import numpy as np
import emout
from vdsolver.core.probs import MaxwellProb

from vdsolver.core.base import (BoundaryList, FieldScalar,
                                SimpleFieldVector3d)
from vdsolver.core.boundaries import (RectangleX, RectangleY, RectangleZ,
                                      create_simbox)
from vdsolver.sims import ESSimulator3d


def create_default_simulator(data: emout.Emout,
                     ispec: int,
                     istep: int = -1,
                     use_si=False,
                     use_hole: bool=None) -> ESSimulator3d:
    # Basic parameters
    nx, ny, nz = data.inp.nx, data.inp.ny, data.inp.nz
    dx = 1.0
    path = data.inp.path[ispec]
    vdri = data.inp.vdri[ispec]

    if use_hole is None:
        use_hole = 'xlrechole' in data.inp

    # Hole parapeters
    if use_hole:
        xl = data.inp.xlrechole[0]
        xu = data.inp.xurechole[0]
        yl = data.inp.xlrechole[0]
        yu = data.inp.yurechole[0]
        zl = data.inp.zlrechole[1]
        zu = data.inp.zurechole[0]

    # Electric field settings
    ex_data = data.ex[istep, :, :, :]
    ey_data = data.ey[istep, :, :, :]
    ez_data = data.ez[istep, :, :, :]

    if use_si:
        dx = data.utit.length.reverse(dx)
        path = data.utit.v.reverse(path)
        vdri = data.unit.v.reverse(vdri)

        if use_hole:
            xl = data.unit.length.reverse(xl)
            xu = data.unit.length.reverse(xu)
            yl = data.unit.length.reverse(yl)
            yu = data.unit.length.reverse(yu)
            zl = data.unit.length.reverse(zl)
            zu = data.unit.length.reverse(zu)

        ex_data = ex_data.val_si
        ey_data = ey_data.val_si
        ez_data = ez_data.val_si

    ex = FieldScalar(ex_data, dx, offsets=(0.5*dx, 0.0, 0.0))
    ey = FieldScalar(ey_data, dx, offsets=(0.0, 0.5*dx, 0.0))
    ez = FieldScalar(ez_data, dx, offsets=(0.0, 0.0, 0.5*dx))
    ef = SimpleFieldVector3d(ex, ey, ez)

    # Velocity distribution
    vdist = MaxwellProb((0, 0, vdri), (path, path, path))
    noprob = NoProb()

    # Boundaries
    boundaries = []

    # Simulation boundary
    simbox = create_simbox(
        xlim=(0.0, nx * dx),
        ylim=(0.0, ny*dx),
        zlim=(0.0, nz*dx),
        func_prob_default=noprob,
        func_prob_dict={
            'zu': vdist,
        },
        use_wall=['zu', 'zl']
    )
    boundaries.append(simbox)

    # Inner boundary
    if use_hole:
        hole = BoundaryList([
            RectangleZ(np.array([0.0, 0.0, zu]), xl, ny*dx, noprob),
            RectangleZ(np.array([xl, 0.0, zu]), xu-xl, yl, noprob),
            RectangleZ(np.array([xu, 0.0, zu]), nx*dx-xu, ny*dx, noprob),
            RectangleZ(np.array([xl, yu, zu]), xu-xl, ny*dx-yu, noprob),
            RectangleX(np.array([xl, yl, zl]), yu-yl, zu-zl, noprob),
            RectangleX(np.array([xu, yl, zl]), yu-yl, zu-zl, noprob),
            RectangleY(np.array([xl, yl, zl]), zu-zl, xu-xl, noprob),
            RectangleY(np.array([xl, yu, zl]), zu-zl, xu-xl, noprob),
            RectangleZ(np.array([xl, yl, zl]), xu-xl, yu-yl, noprob),
        ])
        boundaries.append(hole)

    boundary_list = BoundaryList(boundaries)
    boundary_list.expand()
    sim = ESSimulator3d(nx, ny, nz, dx, ef, boundary_list)
    return sim
