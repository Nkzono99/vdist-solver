from argparse import ArgumentParser
from pathlib import Path


def parse_args_vdsolver():
    parser = ArgumentParser()

    parser.add_argument('axises', default='xz')
    parser.add_argument('--output', '-o', default='vdist-solver.py')
    parser.add_argument('--hybrid', '-hybrid', action='store_true')

    return parser.parse_args()


def gentemp_vdsolver():
    args = parse_args_vdsolver()

    axises: str = args.axises

    if axises == 'current':
        if args.hybrid:
            print('Option \'-hybrid\' is not supported when current is specified')
            return

        gentemp_vdsolver_current(args, 'templates/vdist-solver-current.py.tmp')
        return

    if axises == 'density':
        if args.hybrid:
            print('Option \'-hybrid\' is not supported when density is specified')
            return

        gentemp_vdsolver_density(args, 'templates/vdist-solver-density.py.tmp')
        return

    if axises.startswith('v'):
        chars = [axises[:2], axises[2:]]
    else:
        chars = [axises[:1], axises[1:]]

    if args.hybrid:
        template_filename = 'templates/vdist-solver{dim}d-hybrid.py.tmp'
    else:
        template_filename = 'templates/vdist-solver{dim}d.py.tmp'

    if len(chars) == 1:
        gentemp_vdsolver1d(args, chars, template_filename.format(dim=1))
    elif len(chars) == 2:
        gentemp_vdsolver2d(args, chars, template_filename.format(dim=2))


def gentemp_vdsolver_current(args, template_filepath):
    filepath = Path(__file__).parent.parent / template_filepath
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(text)


def gentemp_vdsolver_density(args, template_filepath):
    filepath = Path(__file__).parent.parent / template_filepath
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(text)


def gentemp_vdsolver1d(args, chars, template_filepath):
    c1,  = chars
    C1 = c1.upper()

    filepath = Path(__file__).parent.parent / template_filepath
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


def gentemp_vdsolver2d(args, chars, template_filepath):
    c1, c2 = chars
    C1, C2 = c1.upper(), c2.upper()

    filepath = Path(__file__).parent.parent / template_filepath
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    axises = ['x', 'y', 'z', 'vx', 'vy', 'vz']

    lim_strs = []
    for axis in axises:
        lim_str = '0' if axis not in chars else f'(-1, 1, N{axis.upper()})'
        lim_strs.append(f'{axis}={lim_str}')
    phase_str = ',\n        '.join(lim_strs)

    shape = []
    for axis in ['z', 'y', 'x', 'vz', 'vy', 'vx']:
        if axis not in chars:
            continue
        shape.append(f'N{axis.upper()}')
    shape_str = ', '.join(shape)

    unit_name1 = 'length' if c1 in ['x', 'y', 'z'] else 'v'
    unit_name2 = 'length' if c2 in ['x', 'y', 'z'] else 'v'

    new = text.format(
        C1=C1,
        C2=C2,
        i1=axises.index(c1),
        i2=axises.index(c2),
        phase=phase_str,
        shape=shape_str,
        unit_name1=unit_name1,
        unit_name2=unit_name2,
    )

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(new)


def parse_args_backtrace():
    parser = ArgumentParser()

    parser.add_argument('axises', default='xz')
    parser.add_argument('--output', '-o', default='backtrace-solver.py')
    parser.add_argument('--hybrid', '-hybrid', action='store_true')

    return parser.parse_args()


def gentemp_backtrace():
    args = parse_args_backtrace()

    chars = list(args.axises)
    c1, c2 = chars
    C1, C2 = c1.upper(), c2.upper()

    if args.hybrid:
        template_filename = 'templates/backtrace-solver-hybrid.py.tmp'
    else:
        template_filename = 'templates/backtrace-solver.py.tmp'

    filepath = Path(__file__).parent.parent / template_filename
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
