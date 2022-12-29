
import emout
import numpy as np
from deprecated import deprecated

from vdsolver.core.base import BoundaryList, FieldScalar, SimpleFieldVector3d
from vdsolver.core.boundaries import (PlaneXY, RectangleX, RectangleY,
                                      RectangleZ, create_simbox)
from vdsolver.core.probs import MaxwellProb, NoProb
from vdsolver.sims.essimulator import ESSimulator3d


def create_innner_boundaries(data: emout.Emout, use_si: bool):
    noprob = NoProb()
    nx, ny, nz = data.inp.nx, data.inp.ny, data.inp.nz
    dx = 1.0

    if use_si:
        dx = data.utit.length.reverse(dx)

    # Inner boundary
    if data.inp.boundary_type == 'rectangle-hole':
        # Hole parapeters
        xl = data.inp.xlrechole[0]
        xu = data.inp.xurechole[0]
        yl = data.inp.xlrechole[0]
        yu = data.inp.yurechole[0]
        zl = data.inp.zlrechole[1]
        zu = data.inp.zurechole[0]

        if use_si:
            xl = data.unit.length.reverse(xl)
            xu = data.unit.length.reverse(xu)
            yl = data.unit.length.reverse(yl)
            yu = data.unit.length.reverse(yu)
            zl = data.unit.length.reverse(zl)
            zu = data.unit.length.reverse(zu)

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

        if use_si:
            zssurf = data.unit.length.reverse(zssurf)

        surf = PlaneXY(zssurf, noprob)
        return surf


def create_emmision_surf(data: emout.Emout, ispec: int, use_si: bool, priority: int = 1):
    path = data.inp.path[ispec]

    if use_si:
        path = data.utit.v.reverse(path)

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

            curf = data.inp.curf[-1]
            try:
                curf = data.inp.curfs[iepl]
            except Exception:
                pass

            pos = np.array([xmine, ymine, zmine])
            xw = xmaxe - xmine
            yw = ymaxe - ymine
            zw = zmaxe - zmine

            if use_si:
                pos = data.unit.length.reverse(pos)
                xw = data.unit.length.reverse(xw)
                yw = data.unit.length.reverse(yw)
                zw = data.unit.length.reverse(zw)
                curf = data.unit.J.reverse(curf)

            n0 = curf / (path / np.sqrt(2*np.pi))
            func_prob = MaxwellProb((0, 0, 0), (path, path, path)) * n0
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
        istep: int = -1,
        use_si=False) -> ESSimulator3d:
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

    if use_si:
        dx = data.utit.length.reverse(dx)
        path = data.utit.v.reverse(path)
        vdri = data.unit.v.reverse(vdri)

    # Electric field settings
    ex_data = data.ex[istep, :, :, :]
    ey_data = data.ey[istep, :, :, :]
    ez_data = data.ez[istep, :, :, :]

    if use_si:
        ex_data = ex_data.val_si
        ey_data = ey_data.val_si
        ez_data = ez_data.val_si

    ex = FieldScalar(ex_data, dx, offsets=(0.5*dx, 0.0, 0.0))
    ey = FieldScalar(ey_data, dx, offsets=(0.0, 0.5*dx, 0.0))
    ez = FieldScalar(ez_data, dx, offsets=(0.0, 0.0, 0.5*dx))
    ef = SimpleFieldVector3d(ex, ey, ez)

    # Magnetic field settings
    try:
        bx_data = data.bx[istep, :, :, :]
        by_data = data.by[istep, :, :, :]
        bz_data = data.bz[istep, :, :, :]
        if use_si:
            bx_data = bx_data.val_si
            by_data = by_data.val_si
            bz_data = bz_data.val_si

        bx = FieldScalar(bx_data, dx, offsets=(0.0, 0.5*dx, 0.5*dx))
        by = FieldScalar(by_data, dx, offsets=(0.5*dx, 0.0, 0.5*dx))
        bz = FieldScalar(bz_data, dx, offsets=(0.5*dx, 0.5*dx, 0.0))
        bf = SimpleFieldVector3d(bx, by, bz)
    except Exception:
        bf = None

    # Velocity distribution
    qe = data.unit.qe.to_unit
    me = data.unit.me.to_unit
    m = me / np.array(data.inp.qm[ispec])
    wp = np.array(data.inp.wp[ispec])
    n0 = np.abs(wp**2*m/qe**2)
    if use_si:
        n0 = data.unit.n.reverse(n0)

    vdist = MaxwellProb((0, 0, vdri), (path, path, path))*n0
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

    ibs = create_innner_boundaries(data, use_si=use_si)
    boundaries.append(ibs)

    # Particle emmition
    pemit = create_emmision_surf(data, ispec, use_si=use_si, priority=1)
    boundaries.append(pemit)

    boundary_list = BoundaryList(boundaries)
    boundary_list.expand()
    sim = ESSimulator3d(nx, ny, nz, dx, ef, bf, boundary_list)
    return sim


@deprecated(reason='Integrated in another function create_default_simulator')
def create_default_pe_simulator(
        data: emout.Emout,
        ispec: int = 2,
        istep: int = -1,
        use_si=False,
        use_hole: bool = None,
        dx: float = 1.0) -> ESSimulator3d:
    # Basic parameters
    nx, ny, nz = data.inp.nx, data.inp.ny, data.inp.nz
    path = data.inp.path[ispec]

    # Electric field settings
    ex_data = data.ex[istep, :, :, :]
    ey_data = data.ey[istep, :, :, :]
    ez_data = data.ez[istep, :, :, :]

    if use_si:
        dx = data.utit.length.reverse(dx)
        path = data.utit.v.reverse(path)

        ex_data = ex_data.val_si
        ey_data = ey_data.val_si
        ez_data = ez_data.val_si

    ex = FieldScalar(ex_data, dx, offsets=(0.5*dx, 0.0, 0.0))
    ey = FieldScalar(ey_data, dx, offsets=(0.0, 0.5*dx, 0.0))
    ez = FieldScalar(ez_data, dx, offsets=(0.0, 0.0, 0.5*dx))
    ef = SimpleFieldVector3d(ex, ey, ez)

    # Velocity distribution
    vdist = MaxwellProb((0, 0, 0), (path, path, path))
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
        },
        use_wall=['zu', 'zl']
    )
    boundaries.append(simbox)

    if use_hole is None:
        use_hole = 'xlrechole' in data.inp

    # Inner boundary
    if use_hole:
        # Hole parapeters
        xl = data.inp.xlrechole[0]
        xu = data.inp.xurechole[0]
        yl = data.inp.xlrechole[0]
        yu = data.inp.yurechole[0]
        zl = data.inp.zlrechole[1]
        zu = data.inp.zurechole[0]

        if use_si:
            xl = data.unit.length.reverse(xl)
            xu = data.unit.length.reverse(xu)
            yl = data.unit.length.reverse(yl)
            yu = data.unit.length.reverse(yu)
            zl = data.unit.length.reverse(zl)
            zu = data.unit.length.reverse(zu)

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
    else:
        zssurf = data.inp.zssurf

        if use_si:
            zssurf = data.unit.length.reverse(zssurf)

        surf = PlaneXY(zssurf, noprob)
        boundaries.append(surf)

    # PE-Emmition
    for iepl in range(data.inp.nepl[-1]):
        nemd = data.inp.nemd[iepl]
        xmine = data.inp.xmine[iepl]
        xmaxe = data.inp.xmaxe[iepl]
        ymine = data.inp.ymine[iepl]
        ymaxe = data.inp.ymaxe[iepl]
        zmine = data.inp.zmine[iepl]
        zmaxe = data.inp.zmaxe[iepl]

        curf = data.inp.curf[-1]
        try:
            curf = data.inp.curfs[iepl]
        except Exception:
            pass

        pos = np.array([xmine, ymine, zmine])
        xw = xmaxe - xmine
        yw = ymaxe - ymine
        zw = zmaxe - zmine

        if use_si:
            pos = data.unit.length.reverse(pos)
            xw = data.unit.length.reverse(xw)
            yw = data.unit.length.reverse(yw)
            zw = data.unit.length.reverse(zw)
            curf = data.unit.J.reverse(curf)

        n_pe = curf / (path / np.sqrt(2*np.pi))
        func_prob = vdist * n_pe
        if abs(nemd) == 1:
            pe_boundary = RectangleX(pos, yw, zw,
                                     func_prob=func_prob,
                                     priority=1)
            boundaries.append(pe_boundary)
        elif abs(nemd) == 2:
            pe_boundary = RectangleY(pos, zw, xw,
                                     func_prob=func_prob,
                                     priority=1)
            boundaries.append(pe_boundary)
        elif abs(nemd) == 3:
            pe_boundary = RectangleZ(pos, xw, yw,
                                     func_prob=func_prob,
                                     priority=1)
            boundaries.append(pe_boundary)

    boundary_list = BoundaryList(boundaries)
    boundary_list.expand()
    sim = ESSimulator3d(nx, ny, nz, dx, ef, boundary_list)
    return sim
