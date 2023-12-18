
import emout
import numpy as np
from deprecated import deprecated
from scipy.spatial.transform import Rotation

from vdsolver.core.base import (Boundary, BoundaryList, FieldScalar,
                                SimpleFieldVector3d)
from vdsolver.core.boundaries import (CircleX, CircleY, CircleZ, CylinderX,
                                      CylinderY, CylinderZ, PlaneXY,
                                      RectangleX, RectangleY, RectangleZ,
                                      Sphere, create_box, create_simbox)
from vdsolver.core.probs import MaxwellProb, NoProb
from vdsolver.sims.essimulator import ESSimulator3d

from typing import Tuple


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
    dx = 1.0

    # Electric field settings
    e0x, e0y, e0z = background_electric_field(data)
    ex_data = data.ex[istep, :, :, :] + e0x
    ey_data = data.ey[istep, :, :, :] + e0y
    ez_data = data.ez[istep, :, :, :] + e0z

    ex = FieldScalar(ex_data, dx, offsets=(0.5*dx, 0.0, 0.0))
    ey = FieldScalar(ey_data, dx, offsets=(0.0, 0.5*dx, 0.0))
    ez = FieldScalar(ez_data, dx, offsets=(0.0, 0.0, 0.5*dx))
    ef = SimpleFieldVector3d(ex, ey, ez)

    # Magnetic field settings
    b0x, b0y, b0z = background_magnetic_field(data)
    try:
        bx_data = data.bx[istep, :, :, :]
        by_data = data.by[istep, :, :, :]
        bz_data = data.bz[istep, :, :, :]

        bx = FieldScalar(bx_data+b0x, dx, offsets=(0.0, 0.5*dx, 0.5*dx))
        by = FieldScalar(by_data+b0y, dx, offsets=(0.5*dx, 0.0, 0.5*dx))
        bz = FieldScalar(bz_data+b0z, dx, offsets=(0.5*dx, 0.5*dx, 0.0))
        bf = SimpleFieldVector3d(bx, by, bz)
    except Exception:
        # If magnetic field data is not available and cannot be loaded,
        # the magnetic field is calculated as 0.
        bx_data = np.zeros_like(ex_data)
        by_data = np.zeros_like(ey_data)
        bz_data = np.zeros_like(ez_data)

        bx = FieldScalar(bx_data+b0x, dx, offsets=(0.0, 0.5*dx, 0.5*dx))
        by = FieldScalar(by_data+b0y, dx, offsets=(0.5*dx, 0.0, 0.5*dx))
        bz = FieldScalar(bz_data+b0z, dx, offsets=(0.5*dx, 0.5*dx, 0.0))

        bf = SimpleFieldVector3d(bx, by, bz)

    boundaries = []

    obs = create_external_boundaries(data, ispec, dx)
    boundaries.append(obs)

    ibs = create_innner_boundaries(data)
    boundaries.append(ibs)

    # Particle emmition
    pemit = create_emmision_surface(data, ispec, priority=1)
    boundaries.append(pemit)

    boundary_list = BoundaryList(boundaries)
    boundary_list.expand()

    nx, ny, nz = data.inp.nx, data.inp.ny, data.inp.nz
    sim = ESSimulator3d(nx, ny, nz, dx, ef, bf, boundary_list)
    return sim


def background_electric_field(data: emout.Emout) -> np.ndarray:
    b0 = background_magnetic_field(data)

    vbulk = bulk_velocity(data, 0)

    return -np.cross(vbulk, b0)


def background_magnetic_field(data: emout.Emout) -> np.ndarray:
    if 'wc' not in data.inp:
        return np.zeros(3)

    b0 = data.inp.wc / data.inp.qm[0]

    return rotate(np.array([0., 0., b0]), data.inp.phiz, data.inp.phixy)


def bulk_velocity(data: emout.Emout, ispec: int) -> np.ndarray:
    vdri = rotate(np.array([0, 0, data.inp.vdri[ispec]]),
                  data.inp.vdthz[ispec], data.inp.vdthxy[ispec])

    if 'spa' in data.inp:
        vdri_respect_to_b0 = vector_with_paraperp(data.inp.spa[ispec],
                                                  data.inp.spe[ispec],
                                                  data.inp.speth[ispec],
                                                  data.inp.phiz,
                                                  data.inp.phixy)
        return vdri + vdri_respect_to_b0

    return vdri


def thermal_velocity(data: emout.Emout, ispec: int) -> np.ndarray:
    para = data.inp.path[ispec]
    perp = data.inp.peth[ispec]
    return rotate([perp, perp, para], data.inp.phiz, data.inp.phixy)


def rotate(vec: np.ndarray, phiz_deg: float, phixy_deg: float) -> np.ndarray:
    rot = Rotation.from_euler('yz', [phiz_deg, phixy_deg], degrees=True)

    return rot.apply(vec)


def vector_with_paraperp(parallel_components: float,
                         perpendicular_component: float,
                         perpendicular_rotate_deg: float,
                         phiz_deg: float,
                         phixy_deg: float) -> np.ndarray:
    rot = Rotation.from_euler('zyz',
                              [perpendicular_rotate_deg, phiz_deg, phixy_deg],
                              degrees=True)

    vec = [perpendicular_component, 0, parallel_components]

    return rot.apply(vec)


def create_external_boundaries(data: emout.Emout, ispec: int, dx: float) -> Boundary:
    if data.inp.nflag_emit[ispec] == 2:
        return BoundaryList([])

    # Velocity distribution
    wall_prob_dict = {}
    vbulk = bulk_velocity(data, ispec)
    vthermal = thermal_velocity(data, ispec)
    vdist = MaxwellProb(vbulk, vthermal)

    if data.inp.npbnd[3*ispec + 0] == 2:  # X boundary
        wall_prob_dict['xl'] = vdist
        wall_prob_dict['xu'] = vdist

    if data.inp.npbnd[3*ispec + 1] == 2:  # Y boundary
        wall_prob_dict['yl'] = vdist
        wall_prob_dict['yu'] = vdist

    if data.inp.npbnd[3*ispec + 2] == 2:  # Z boundary
        if 'zssurf' not in data.inp or data.inp.zssurf < 0:
            wall_prob_dict['zl'] = vdist
        wall_prob_dict['zu'] = vdist

    # Simulation boundary
    nx, ny, nz = data.inp.nx, data.inp.ny, data.inp.nz
    simbox = create_simbox(
        xlim=(0.0, nx*dx),
        ylim=(0.0, ny*dx),
        zlim=(0.0, nz*dx),
        func_prob_default=NoProb(),
        func_prob_dict=wall_prob_dict,
        priority_prob_dict={key: 0. for key in wall_prob_dict},
        use_wall=wall_prob_dict.keys()
    )

    return simbox


def create_innner_boundaries(data: emout.Emout) -> Boundary:
    noprob = NoProb()
    nx, ny, nz = data.inp.nx, data.inp.ny, data.inp.nz
    dx = 1.0

    boundaries = []

    # Setting of internal boundaries using geotype
    geotypes = create_innner_boundaries_from_geotypes(data)
    boundaries.append(geotypes)

    # Setting of internal boundaries not set by geotype
    if 'boundary_type' in data.inp:
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
            boundaries.append(hole)
        elif data.inp.boundary_type == 'flat-surface':
            zssurf = data.inp.zssurf

            surf = PlaneXY(zssurf, noprob)
            boundaries.append(surf)
        else:
            raise NotImplementedError()

    elif 'zssurf' in data.inp:
        zssurf = data.inp.zssurf

        surf = PlaneXY(zssurf, noprob)
        boundaries.append(surf)

    return BoundaryList(boundaries)


def create_innner_boundaries_from_geotypes(data: emout.Emout) -> Boundary:
    boundaries = []

    if 'npc' not in data.inp:
        return BoundaryList(boundaries)

    for ipc in range(data.inp.npc):
        if data.inp.geotype[ipc] in (0, 1):
            rect = create_rectangular_boundary(data, ipc)
            boundaries.append(rect)
        elif data.inp.geotype[ipc] == 2:
            cylinder = create_cylinder_boundary(data, ipc)
            boundaries.append(cylinder)
        elif data.inp.geotype[ipc] == 3:
            sphere = create_sphere_boundary(data, ipc)
            boundaries.append(sphere)
        else:
            raise NotImplementedError()

    return BoundaryList(boundaries)


def create_rectangular_boundary(data: emout.Emout, ipc: int) -> Boundary:
    xlpc = fetch_from_inp(data, 'ptcond', 'xlpc', default=0.)
    ylpc = fetch_from_inp(data, 'ptcond', 'ylpc', default=0.)
    zlpc = fetch_from_inp(data, 'ptcond', 'zlpc', default=0.)
    xupc = fetch_from_inp(data, 'ptcond', 'xupc', default=0.)
    yupc = fetch_from_inp(data, 'ptcond', 'yupc', default=0.)
    zupc = fetch_from_inp(data, 'ptcond', 'zupc', default=0.)

    return create_box(
        (xlpc[ipc], xupc[ipc]),
        (ylpc[ipc], yupc[ipc]),
        (zlpc[ipc], zupc[ipc]),
        use_wall='all',
    )


def create_cylinder_boundary(data: emout.Emout, ipc: int) -> Boundary:
    bdyalign = fetch_from_inp(data, 'ptcond', 'bdyalign', default=0)
    bdyedge = fetch_from_inp(data, 'ptcond', 'bdyedge', default=0)
    bdyradius = fetch_from_inp(data, 'ptcond', 'bdyradius', default=0)
    bdycoord = fetch_from_inp(data, 'ptcond', 'bdycoord', default=0)

    height = bdyedge[ipc, 1] - bdyedge[ipc, 0]
    radius = bdyradius[ipc]
    if bdyalign[ipc] == 1:
        origin = \
            np.array([bdyedge[ipc, 0], bdycoord[ipc, 0], bdycoord[ipc, 1]])
        upper_origin = origin + np.array([height, 0, 0])

        cylinder = CylinderX(origin, radius, height, func_prob=NoProb())
        circle_lower = CircleX(origin, radius, func_prob=NoProb())
        circle_upper = CircleX(upper_origin, radius, func_prob=NoProb())

        return BoundaryList([cylinder, circle_lower, circle_upper])

    if bdyalign[ipc] == 2:
        origin = \
            np.array([bdycoord[ipc, 1], bdyedge[ipc, 0], bdycoord[ipc, 0]])
        upper_origin = origin + np.array([0, height, 0])

        cylinder = CylinderY(origin, radius, height, func_prob=NoProb())
        circle_lower = CircleY(origin, radius, func_prob=NoProb())
        circle_upper = CircleY(upper_origin, radius, func_prob=NoProb())

        return BoundaryList([cylinder, circle_lower, circle_upper])

    if bdyalign[ipc] == 3:
        origin = \
            np.array([bdycoord[ipc, 0], bdycoord[ipc, 1], bdyedge[ipc, 0]])
        upper_origin = origin + np.array([0, 0, height])

        cylinder = CylinderZ(origin, radius, height, func_prob=NoProb())
        circle_lower = CircleZ(origin, radius, func_prob=NoProb())
        circle_upper = CircleZ(upper_origin, radius, func_prob=NoProb())

        return BoundaryList([cylinder, circle_lower, circle_upper])


def create_sphere_boundary(data: emout.Emout, ipc: int) -> Boundary:
    bdyradius = fetch_from_inp(data, 'ptcond', 'bdyradius', default=0.)
    bdycoord = fetch_from_inp(data, 'ptcond', 'bdycoord', default=0.)

    return Sphere(bdycoord[ipc, :], bdyradius[ipc], func_prob=NoProb())


def create_emmision_surface(data: emout.Emout, ispec: int, priority: int = 1) -> Boundary:
    # Settings using 'nepl'.
    if 'nepl' not in data.inp:
        return BoundaryList([])

    if data.inp.nepl == 0:
        return BoundaryList([])

    boundaries = []

    nepls = fetch_from_inp(data, 'emissn', 'nepl', default=0)

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

        vbulk = bulk_velocity_emission(data, ispec, nemd)
        vthermal = thermal_velocity_emission(data, ispec, nemd)
        func_prob = MaxwellProb(vbulk, vthermal)

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

    return BoundaryList(boundaries)


def bulk_velocity_emission(data, ispec, nemd):
    vdri = rotate(data.inp.vdri[ispec],
                  data.inp.vdthz[ispec], data.inp.vdthxy[ispec])

    if 'spa' in data.inp:
        phiz, phixy = to_phixyz(nemd)

        vdri_respect_to_b0 = vector_with_paraperp(data.inp.spa[ispec],
                                                  data.inp.spe[ispec],
                                                  data.inp.speth[ispec],
                                                  phiz,
                                                  phixy)
        return vdri + vdri_respect_to_b0

    return vdri


def thermal_velocity_emission(data: emout.Emout, ispec: int, nemd: int) -> np.ndarray:
    para = data.inp.path[ispec]
    perp = data.inp.peth[ispec]

    phiz, phixy = to_phixyz(nemd)

    return rotate([perp, perp, para], phiz, phixy)


def to_phixyz(nemd: int) -> Tuple[int, int]:
    if abs(nemd) == 1:
        phiz = 90
        phixy = 0
    elif abs(nemd) == 2:
        phiz = 90
        phixy = 90
    elif abs(nemd) == 3:
        phiz = 0
        phixy = 0

    if nemd < 0:
        phiz -= 180

    return phiz, phixy


def fetch_from_inp(data: emout.Emout, group: str, name: str, default=0) -> np.ndarray:
    start_index = np.array(data.inp.nml[group].start_index[name][::-1]) - 1
    values = np.array(getattr(data.inp, name))
    original_shape = start_index + np.array(values.shape)

    original_values = np.full(original_shape, default)
    original_values[tuple(slice(i, None) for i in start_index)] = values

    return original_values
