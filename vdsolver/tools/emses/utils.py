
import emout
import numpy as np
from deprecated import deprecated

from vdsolver.core.base import BoundaryList, FieldScalar, SimpleFieldVector3d
from vdsolver.core.boundaries import (PlaneXY, RectangleX, RectangleY,
                                      RectangleZ, create_simbox)
from vdsolver.core.probs import MaxwellProb, NoProb
from vdsolver.sims.essimulator import ESSimulator3d


def create_innner_boundaries(data: emout.Emout):
    noprob = NoProb()
    nx, ny, nz = data.inp.nx, data.inp.ny, data.inp.nz
    dx = 1.0

    # Inner boundary
    if data.inp.boundary_type == 'rectangle-hole':
        # Hole parapeters
        xl = data.inp.xlrechole[0]
        xu = data.inp.xurechole[0]
        yl = data.inp.xlrechole[0]
        yu = data.inp.yurechole[0]
        zl = data.inp.zlrechole[1]
        zu = data.inp.zurechole[0]

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

        return hole
    else:
        zssurf = data.inp.zssurf

        surf = PlaneXY(zssurf, noprob)
        return surf


def create_emmision_surf(data: emout.Emout, ispec: int, priority: int = 1):
    path = data.inp.path[ispec]

    boundaries = []

    try:
        start_index = data.inp.nml['emissn'].start_index['nepl'][0]
        nepls = []
        for _ in range(start_index-1):
            nepls.append(0)
        for nepl in data.inp.nepl:
            nepls.append(nepl)

        iepl_start = int(np.sum(nepls[:ispec]))
        iepl_end = int(np.sum(nepls[:ispec+1]))
        for iepl in range(iepl_start, iepl_end):
            nemd = data.inp.nemd[iepl]
            xmine = data.inp.xmine[iepl]
            xmaxe = data.inp.xmaxe[iepl]
            ymine = data.inp.ymine[iepl]
            ymaxe = data.inp.ymaxe[iepl]
            zmine = data.inp.zmine[iepl]
            zmaxe = data.inp.zmaxe[iepl]

            pos = np.array([xmine, ymine, zmine])
            xw = xmaxe - xmine
            yw = ymaxe - ymine
            zw = zmaxe - zmine

            func_prob = MaxwellProb((0, 0, 0), (path, path, path))
            if abs(nemd) == 1:
                boundary = RectangleX(pos, yw, zw,
                                      func_prob=func_prob,
                                      priority=priority)
                boundaries.append(boundary)
            elif abs(nemd) == 2:
                boundary = RectangleY(pos, zw, xw,
                                      func_prob=func_prob,
                                      priority=priority)
                boundaries.append(boundary)
            elif abs(nemd) == 3:
                boundary = RectangleZ(pos, xw, yw,
                                      func_prob=func_prob,
                                      priority=priority)
                boundaries.append(boundary)
    except AttributeError:
        pass

    return BoundaryList(boundaries)


def create_default_simulator(
        data: emout.Emout,
        ispec: int,
        istep: int = -1) -> ESSimulator3d:
    """Create 3D Electric Static Simulator with reference to the the parameter file for EMSES (e.g. 'plasma.inp').

    Parameters
    ----------
    data : emout.Emout
        Simulation data for EMSES.
    ispec : int
        Particle species number (typically 0: electron, 1: ion, 2: photoelectron).
    istep : int, optional
        Output step for simulation data to be used, by default -1 (= final output data).
    use_si : bool, optional
        True for calculations in the SI system of units, by default False.

    Returns
    -------
    ESSimulator3d
        3D Electric Static Simulator with reference to the the parameter file for EMSES
    """
    # Basic parameters
    nx, ny, nz = data.inp.nx, data.inp.ny, data.inp.nz
    dx = 1.0
    path = data.inp.path[ispec]
    vdri = data.inp.vdri[ispec]

    # Electric field settings
    ex_data = data.ex[istep, :, :, :]
    ey_data = data.ey[istep, :, :, :]
    ez_data = data.ez[istep, :, :, :]

    ex = FieldScalar(ex_data, dx, offsets=(0.5*dx, 0.0, 0.0))
    ey = FieldScalar(ey_data, dx, offsets=(0.0, 0.5*dx, 0.0))
    ez = FieldScalar(ez_data, dx, offsets=(0.0, 0.0, 0.5*dx))
    ef = SimpleFieldVector3d(ex, ey, ez)

    # Magnetic field settings
    try:
        bx_data = data.bx[istep, :, :, :]
        by_data = data.by[istep, :, :, :]
        bz_data = data.bz[istep, :, :, :]

        bx = FieldScalar(bx_data, dx, offsets=(0.0, 0.5*dx, 0.5*dx))
        by = FieldScalar(by_data, dx, offsets=(0.5*dx, 0.0, 0.5*dx))
        bz = FieldScalar(bz_data, dx, offsets=(0.5*dx, 0.5*dx, 0.0))
        bf = SimpleFieldVector3d(bx, by, bz)
    except Exception:
        bf = None

    # Velocity distribution
    vbulk = (vdri*np.sin(np.deg2rad(data.inp.vdthz[ispec]))*np.cos(np.deg2rad(data.inp.vdthxy[ispec])),
             vdri*np.sin(np.deg2rad(data.inp.vdthz[ispec]))*np.sin(np.deg2rad(data.inp.vdthxy[ispec])),
             vdri*np.cos(np.deg2rad(data.inp.vdthz[ispec])))

    vdist = MaxwellProb(vbulk, (path, path, path))
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
        priority_prob_dict={
            'zu': 0,
        },
        use_wall=['zu', 'zl']
    )
    boundaries.append(simbox)

    ibs = create_innner_boundaries(data)
    boundaries.append(ibs)

    # Particle emmition
    pemit = create_emmision_surf(data, ispec, priority=1)
    boundaries.append(pemit)

    boundary_list = BoundaryList(boundaries)
    boundary_list.expand()
    sim = ESSimulator3d(nx, ny, nz, dx, ef, bf, boundary_list)
    return sim
