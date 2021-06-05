from argparse import ArgumentParser
from collections import deque
from typing import Deque, List
from vdsolver.tools.plot import plot_periodic

import numpy as np
import emout
import matplotlib.pyplot as plt
import scipy.constants as cn
from scipy.stats import norm
from tqdm import tqdm

from vdsolver.base import (BoundaryList, FieldScalar, Particle,
                           SimpleFieldVector3d)
from vdsolver.boundaries import (RectangleX, RectangleY, RectangleZ,
                                 create_simbox)
from vdsolver.emsolver import ChargedParticle, ESSimulator3d


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--directory', '-d')
    parser.add_argument('--istep', '-is', default=-1, type=int)
    parser.add_argument('--maxstep', '-ms', default=10000, type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    data = emout.Emout(args.directory)

    nx, ny, nz = data.inp.nx, data.inp.ny, data.inp.nz
    dx = 1.0
    ex_data = data.ex[args.istep, :, :, :]
    ey_data = data.ey[args.istep, :, :, :]
    ez_data = data.ez[args.istep, :, :, :]

    dt = data.inp.dt * 0.1

    # q_m = -cn.e / cn.m_e
    q_m = -1.0

    ex = FieldScalar(ex_data, dx, offsets=(0.5*dx, 0.0, 0.0))
    ey = FieldScalar(ey_data, dx, offsets=(0.0, 0.5*dx, 0.0))
    ez = FieldScalar(ez_data, dx, offsets=(0.0, 0.0, 0.5*dx))
    ef = SimpleFieldVector3d(ex, ey, ez)

    def vdist(vel):
        px = norm.pdf(vel[0], loc=0.0, scale=data.inp.path[0])
        py = norm.pdf(vel[1], loc=0.0, scale=data.inp.path[0])
        pz = norm.pdf(vel[2], loc=data.inp.vdri[0], scale=data.inp.path[0])
        return px * py * pz

    simbox = create_simbox(
        xlim=(0.0, nx * dx),
        ylim=(0.0, ny*dx),
        zlim=(0.0, nz*dx),
        func_prob_default=lambda vel: 0.0,
        func_prob_dict={
            'zu': vdist,
        },
        use_wall=['zu', 'zl']
    )

    xl = data.inp.xlrechole[0]
    xu = data.inp.xurechole[0]
    yl = data.inp.xlrechole[0]
    yu = data.inp.yurechole[0]
    zl = data.inp.zlrechole[1]
    zu = data.inp.zurechole[0]

    def noprob(vel): return 0.0
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

    boundary_list = BoundaryList([simbox, hole])
    boundary_list.expand()
    sim = ESSimulator3d(nx, ny, nz, dx, ef, boundary_list)

    pos = np.array([13.0, 16.0, 55.0])
    vx = np.linspace(-100, 100, 10)
    vz = np.linspace(-200, 0, 200)
    VX, VZ = np.meshgrid(vx, vz)
    vels = np.zeros((len(vz), len(vx), 3))
    vels[:, :, 0] = VX
    vels[:, :, 2] = VZ
    # vels = np.zeros((len(vz), 3))
    # vels[:, 2] = vz
    probs = sim.get_probs(pos, vels, q_m, dt,
                          max_step=args.maxstep, show_progress=True)

    # plt.plot(vz, probs)
    # plt.grid()

    plt.pcolormesh(VX, VZ, probs)
    plt.colorbar()

    plt.show()

    # vzs = np.linspace(-200, 30, 100)
    # for vz in tqdm(vzs):
    #     pos = np.array([5.0, 5.0, 61.0])
    #     vel = np.array([0.0, 0.0, vz])
    #     pcl = ChargedParticle(pos, vel, q_m)
    #     prob, pcl_last = sim.get_prob(
    #         pcl, dt=dt, max_step=args.maxstep)
    #     probs.append(prob)

    # pos = np.array([16.0, 16.0, 21.0])
    # vel = np.array([-40.0, 0.0, 1000.0])
    # pcl = ChargedParticle(pos, vel, q_m)

    # history: Deque[ChargedParticle] = deque()
    # prob, pcl_last = sim.get_prob(
    #     pcl, dt=dt, max_step=args.maxstep, history=history)
    # print('Probabirity of {pcl}: {prob}'.format(pcl=pcl_last, prob=prob))

    # plt.subplot(1, 2, 1)
    # ez_data[:, 16, :].plot(use_si=False)
    # plot_periodic(history)
    # plt.xlim([0, 32])
    # plt.ylim([0, 252])
    # plt.subplot(1, 2, 2)
    # ez_data[:, :, 16].plot(use_si=False)
    # plot_periodic(history, idxs=[1, 2])
    # plt.xlim([0, 32])
    # plt.ylim([0, 252])
    # plt.show()


if __name__ == '__main__':
    main()
