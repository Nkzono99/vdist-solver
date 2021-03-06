from argparse import ArgumentParser
from pathlib import Path


def parse_args_vdsolver():
    parser = ArgumentParser()

    parser.add_argument('axises', default='xz')
    parser.add_argument('--output', '-o', default='vdist-solver.py')

    return parser.parse_args()


def gentemp_vdsolver():
    args = parse_args_vdsolver()

    axises: str = args.axises
    if axises.startswith('v'):
        chars = [axises[:2], axises[2:]]
    else:
        chars = [axises[:1], axises[1:]]

    if len(chars) == 1:
        gentemp_vdsolver1d(args, chars)
    elif len(chars) == 2:
        gentemp_vdsolver2d(args, chars)


def gentemp_vdsolver1d(args, chars):
    c1,  = chars
    C1 = c1.upper()

    filepath = Path(__file__).parent.parent.parent / \
        'templates/vdist-solver1d.py.tmp'
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    axises = ['x', 'y', 'z', 'vx', 'vy', 'vz']

    lim_strs = []
    for axis in axises:
        lim_str = '0' if axis not in chars else f'(-1, 1, N{axis.upper()})'
        lim_strs.append(f'{axis}={lim_str}')
    phase_str = ',\n        '.join(lim_strs)

    new = text.format(
        C1=C1,
        i1=axises.index(c1),
        phase=phase_str,
    )

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(new)


def gentemp_vdsolver2d(args, chars):
    c1, c2 = chars
    C1, C2 = c1.upper(), c2.upper()

    filepath = Path(__file__).parent.parent.parent / \
        'templates/vdist-solver2d.py.tmp'
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    axises = ['x', 'y', 'z', 'vx', 'vy', 'vz']

    lim_strs = []
    for axis in axises:
        lim_str = '0' if axis not in chars else f'(-1, 1, N{axis.upper()})'
        lim_strs.append(f'{axis}={lim_str}')
    phase_str = ',\n        '.join(lim_strs)

    new = text.format(
        C1=C1,
        C2=C2,
        i1=axises.index(c1),
        i2=axises.index(c2),
        phase=phase_str,
    )

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(new)


def parse_args_backtrace():
    parser = ArgumentParser()

    parser.add_argument('axises', default='xz')
    parser.add_argument('--output', '-o', default='backtrace-solver.py')

    return parser.parse_args()


def gentemp_backtrace():
    args = parse_args_backtrace()

    chars = list(args.axises)
    c1, c2 = chars
    C1, C2 = c1.upper(), c2.upper()

    filepath = Path(__file__).parent.parent.parent / \
        'templates/backtrace-solver.py.tmp'
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    new = text.format(
        C1=C1,
        C2=C2,
        i1=['x', 'y', 'z'].index(c1),
        i2=['x', 'y', 'z'].index(c2),
    )

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(new)
