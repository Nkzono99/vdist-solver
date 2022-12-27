from pathlib import Path

from vdsolver.core import FieldScalar, SimpleFieldVector3d
from vdsolver.core.base import BoundaryList
from vdsolver.core.boundaries import create_simbox
from vdsolver.core.probs import MaxwellProb, NoProb
from vdsolver.sims.essimulator import ESSimulator3d

from .data import HybridData


def create_simulation_for_hybrid(
        dirpath: Path,
        dx,
        vth,
        vdri,
        index,
        n0=1.0,
        coef_ef=1.0,
        coef_bf=1.0):

    ex_data = HybridData(dirpath/f'propxyz_EX{index:05d}.vts').values*coef_ef
    ey_data = HybridData(dirpath/f'propxyz_EY{index:05d}.vts').values*coef_ef
    ez_data = HybridData(dirpath/f'propxyz_EZ{index:05d}.vts').values*coef_ef

    ex = FieldScalar(ex_data, dx, offsets=(0.5*dx, 0.0, 0.0))
    ey = FieldScalar(ey_data, dx, offsets=(0.0, 0.5*dx, 0.0))
    ez = FieldScalar(ez_data, dx, offsets=(0.0, 0.0, 0.5*dx))
    ef = SimpleFieldVector3d(ex, ey, ez)

    bx_data = HybridData(dirpath/f'propxyz_BX{index:05d}.vts').values*coef_bf
    by_data = HybridData(dirpath/f'propxyz_BY{index:05d}.vts').values*coef_bf
    bz_data = HybridData(dirpath/f'propxyz_BZ{index:05d}.vts').values*coef_bf

    bx = FieldScalar(bx_data, dx, offsets=(0.0, 0.5*dx, 0.5*dx))
    by = FieldScalar(by_data, dx, offsets=(0.5*dx, 0.0, 0.5*dx))
    bz = FieldScalar(bz_data, dx, offsets=(0.5*dx, 0.5*dx, 0.0))
    bf = SimpleFieldVector3d(bx, by, bz)

    nx, ny, nz = ex_data.shape

    vdist = MaxwellProb((vdri, 0, 0), (vth, vth, vth))*n0
    noprob = NoProb()

    # Boundaries
    boundaries = []

    # Simulation boundary
    simbox = create_simbox(
        xlim=(0.0, nx*dx),
        ylim=(0.0, ny*dx),
        zlim=(0.0, nz*dx),
        func_prob_default=noprob,
        func_prob_dict={
            'xl': vdist,
        },
        priority_prob_dict={
            'xl': 0,
        },
        use_wall=['xl', 'xu']
    )
    boundaries.append(simbox)

    boundary_list = BoundaryList(boundaries)
    boundary_list.expand()
    sim = ESSimulator3d(nx, ny, nz, dx, ef, bf, boundary_list)

    return sim
