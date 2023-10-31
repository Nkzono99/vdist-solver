# 速度分布関数ソルバーの使い方



vdist-solverは、Pythonで粒子軌道及び任意の位相密度分布を取得するためのツールです。

|vx-vz 速度分布(2次元)|vz 速度分布(1次元)|粒子軌道(バックトレース)|
|---|---|---|
|![16_16_65](https://user-images.githubusercontent.com/71783375/120922088-2df7ab00-c702-11eb-876b-1b4538c2c9ad.png)|![vdist_ion_z](https://user-images.githubusercontent.com/71783375/120922107-3fd94e00-c702-11eb-901a-32576eccf53b.png)|![backtrace_test](https://user-images.githubusercontent.com/71783375/120922700-a0b65580-c705-11eb-9410-85b841e5718d.png)|

現在、研究室で開発されているEMSESとHybrid Simulatorをサポートしています。

ただし、すべてのパラメータをサポートしているわけではないため、本ツールが想定していないシミュレーションモデルでは、そのモデルに適用させるためのスクリプトの追加を行う必要があります (参照: [EMSESでのシミュレーションモデルの作成コード](https://github.com/Nkzono99/vdist-solver/blob/main/vdsolver/tools/emses/utils.py))。


## 目次

- [速度分布関数ソルバーの使い方](#速度分布関数ソルバーの使い方)
  - [目次](#目次)
  - [対応しているシミュレーションモデル(2023/11/1 現在)](#対応しているシミュレーションモデル2023111-現在)
    - [EMSES](#emses)
    - [Hybrid Simulator](#hybrid-simulator)
  - [インストール方法](#インストール方法)
  - [EMSESでの速度分布関数ソルバーの使い方](#emsesでの速度分布関数ソルバーの使い方)
    - [粒子軌道の取得方法 (バックトレース/フォワードトレース)](#粒子軌道の取得方法-バックトレースフォワードトレース)
      - [1. スクリプトの雛形の生成](#1-スクリプトの雛形の生成)
      - [2. トレースする粒子の位置・速度の設定](#2-トレースする粒子の位置速度の設定)
      - [3. 実行](#3-実行)
    - [速度分布の取得方法](#速度分布の取得方法)
      - [1. スクリプトの雛形の生成](#1-スクリプトの雛形の生成-1)
      - [2. 取得する速度分布の範囲や位置の設定](#2-取得する速度分布の範囲や位置の設定)
      - [3. 実行](#3-実行-1)


## 対応しているシミュレーションモデル(2023/11/1 現在)

### EMSES
- +z方向のxy平面からのShifted Maxwellian分布に従う太陽風による月面帯電モデル
- -z方向のxy平面からのShifted Maxwellian分布に従う太陽風による月面空洞帯電モデル

### Hybrid Simulator
- -x方向のyz平面方向からのShifted Maxwellian分布に従う太陽風モデル

## インストール方法

以下のコマンドを実行する。

注意: Python3.7系未満での動作はテストしていません。 もし上手く動作しなければ、pythonのバージョンをアップグレードしてください。

```
pip install git+https://github.com/Nkzono99/vdist-solver.git
```


## EMSESでの速度分布関数ソルバーの使い方

### 粒子軌道の取得方法 (バックトレース/フォワードトレース)

#### 1. スクリプトの雛形の生成

まず、以下のコマンドを実行しスクリプトファイル(xy平面への投射図をプロットする)の雛形を生成します。

```
gen-backtrace xy -o backtrace-solver.py
```

#### 2. トレースする粒子の位置・速度の設定

```##```のコメントをつけている行を修正して、粒子軌道を得る粒子の座標・速度(EMSES単位系)を設定してください。

注意: プロット範囲がデフォルトのままだと上手く行かない可能性があるため、```plt.ylabel('Y')``` 前後に ```plt.xlim([0, 32]); plt.ylim([0, 256])``` などを追加してください。

``` python {.line-numbers}
[backtrace-solver.py]

"""Calculate and plot particle orbit using the backtrace method.

    How to run
    ----------
    $ python backtrace-solver.py -d <directory> -index <index> -i <ispec> -o output.png
"""
from argparse import ArgumentParser

import emout
import matplotlib.pyplot as plt
import numpy as np

from vdsolver.core import BackTraceTarget, plot_periodic
from vdsolver.sims.essimulator import ChargedParticle
from vdsolver.tools.emses import create_default_simulator


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--directory', '-d', default='./')

    parser.add_argument('--istep', '-is', default=-1, type=int)
    parser.add_argument('--ispec', '-i', default=0, type=int)

    parser.add_argument('--dt', '-dt', default=1.0, type=float)
    parser.add_argument('--maxstep', '-ms', default=10000, type=int)

    parser.add_argument('--use_si', action='store_true')

    parser.add_argument('--output', '-o', default=None)

    return parser.parse_args()


def create_simulator(*args, **kwargs):
    sim = None
    return sim


def main():
    args = parse_args()

    data = emout.Emout(args.directory)

    pos = np.array([0, 0, 0]) ## 粒子の設定座標
    vel = np.array([0, 0, 0]) ## 粒子の設定速度

    # Create simulator
    sim = create_default_simulator(data, args.ispec, args.istep,
                                   use_si=False)
    # sim = create_simulator()

    pcl_prototype = ChargedParticle.create_prototype(q_m=data.inp.qm[args.ispec])
    target = BackTraceTarget(sim,
                             pcl_prototype,
                             data.inp.dt*args.dt,
                             pos,
                             vel,
                             args.maxstep,
                             )

    history, prob, pcl_last = target.solve()

    if args.use_si:
        for pcl in history:
            pcl.pos = data.unit.length.reverse(pcl.pos)
            pcl.vel = data.unit.v.reverse(pcl.vel)
            pcl.t = data.unit.t.reverse(pcl.t)

    plot_periodic(history, idxs=[0, 1])
    plt.xlabel('X')
    plt.ylabel('Y')

    if args.output is None:
        plt.show()
    else:
        plt.gcf().savefig(args.output)


if __name__ == '__main__':
    main()
```

#### 3. 実行

以下を実行すると粒子軌道がプロットされ、```trace.png```に保存されます。

```
python backtrace-solver.py -is <istep> -i <ispec> -dt <dt> -o trace.png

<istep>: 出力ステップのインデックス(指定しない場合は-1が設定される。-1の場合最終出力ステップの電磁場を用いる)
<ispec>: 粒子種番号 (0: 電子, 1: イオン, 2: 光電子)
<dt>: シミュレーションで用いた時間幅の何倍の時間幅を粒子トレースに用いるか。(デフォルトでは1が設定される。簡単に速度分布を見積る際やイオンなど速度の小さい粒子種の速度分布を取得する際は、ある程度大きくすると効率的に考察を進めることができる。)
```

### 速度分布の取得方法

#### 1. スクリプトの雛形の生成

まず、以下のコマンドを実行しスクリプトファイル(vxvy平面の速度分布を取得)の雛形を生成します。

```
gen-vdsolver vxvy -o vdist-solver.py
```

#### 2. 取得する速度分布の範囲や位置の設定

```##```のコメントをつけている行を修正して、取得したい速度分布の座標・速度(EMSES単位系)を設定してください。

注意: 速度の範囲の設定は各自が決める恣意的なパラメータのため、実際の速度分布を完全に被覆できるかを事前に知ることはできません。そのため、粒子の熱速度・バルク速度、設定した位置のポテンシャルなどのシミュレーションパラメータから事前に速度範囲を推定し、その後いくつか試行錯誤することをおすすめします。

``` python {.line-numbers}
[vdist-solver.py]

"""Calculate and plot velocity distributions using the backtrace method.

    How to run
    ----------
    $ python vdist-solver2d.py -d <directory> -i <ispec> -o output.png
"""
from argparse import ArgumentParser

import emout
import matplotlib.pyplot as plt
import numpy as np

from vdsolver.core import PhaseGrid, VSolveTarget
from vdsolver.sims.essimulator import ChargedParticle
from vdsolver.tools.emses import create_default_simulator


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--directory', '-d', default='./')
    parser.add_argument('--istep', '-is', default=-1, type=int)

    parser.add_argument('--ispec', '-i', default=0, type=int)

    parser.add_argument('--dt', '-dt', default=1.0, type=float)
    parser.add_argument('--maxstep', '-ms', default=10000, type=int)

    parser.add_argument('--use_si', action='store_true')

    parser.add_argument('--output', '-o', default=None)

    parser.add_argument('--max_workers', '-mw', default=8, type=int)
    parser.add_argument('--chunksize', '-chk', default=100, type=int)

    parser.add_argument('--use_mpi', '-mpi', action='store_true')

    return parser.parse_args()


def create_simulator(*args, **kwargs):
    sim = None
    return sim


def main():
    args = parse_args()

    data = emout.Emout(args.directory)

    NVX = 100 ## 取得する速度分布のvx方向のグリッド数
    NVY = 100 ## 取得する速度分布のvy方向のグリッド数

    phase_grid = PhaseGrid(
        x=0, ## 速度分布を取得する位置 (x座標)
        y=0, ## 速度分布を取得する位置 (y座標)
        z=0, ## 速度分布を取得する位置 (z座標)
        vx=(-1, 1, NVX), ## 速度分布を取得する速度範囲 (x軸方向の速度)
        vy=(-1, 1, NVY), ## 速度分布を取得する速度範囲 (y軸方向の速度)
        vz=0 ## 速度分布を取得する速度 (z軸方向の速度)
    )

    # Create simulator
    sim = create_default_simulator(data,
                                   args.ispec,
                                   args.istep,
                                   use_si=False)

    # For self-simulation
    # sim = create_simulator()

    pcl_prototype = ChargedParticle.create_prototype(q_m=data.inp.qm[args.ispec])
    target = VSolveTarget(sim,
                          pcl_prototype,
                          data.inp.dt*args.dt,
                          phase_grid,
                          args.maxstep,
                          args.max_workers,
                          args.chunksize,
                          show_progress=True,
                          use_mpi=args.use_mpi,
                          )

    phases, probs = target.solve()

    phases = phases.reshape(NVY, NVX, 6)
    VX = phases[:, :, 3]
    VY = phases[:, :, 4]
    if args.use_si:
        VX = data.unit.v.reverse(VX)
        VY = data.unit.v.reverse(VY)
    probs = probs.reshape(NVY, NVX)

    plt.pcolormesh(VX, VY, probs, shading='auto')
    plt.colorbar()
    plt.xlabel('VX')
    plt.ylabel('VY')

    if args.output is None:
        plt.show()
    else:
        plt.gcf().savefig(args.output)


if __name__ == '__main__':
    main()
```

#### 3. 実行

以下を実行すると粒子軌道がプロットされ、```vdsit.png```に保存されます。

```
python vdist-solver.py -is <istep> -i <ispec> -dt <dt> -mw <max_workers> -o vdist.png

<istep>: 出力ステップのインデックス(指定しない場合は-1が設定される。-1の場合最終出力ステップの電磁場を用いる)
<ispec>: 粒子種番号 (0: 電子, 1: イオン, 2: 光電子)
<dt>: シミュレーションで用いた時間幅の何倍の時間幅を粒子トレースに用いるか。(デフォルトでは1が設定される。簡単に速度分布を見積る際やイオンなど速度の小さい粒子種の速度分布を取得する際は、ある程度大きくすると効率的に考察を進めることができる。)
<max_workers>: プロセス並列数
```

