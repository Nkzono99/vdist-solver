from argparse import ArgumentParser
from pathlib import Path


def parse_args_vdsolver():
    parser = ArgumentParser()

    parser.add_argument('axises', default='xz')
    parser.add_argument('--output', '-o', default='vdist-solver.py')

    return parser.parse_args()


def gentemp_vdsolver():
    args = parse_args_vdsolver()
    chars = list(args.axises)
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

    line = 'v{}lim = {}'
    lim_str = line.format(
        'x', '[-1.0, 1.0, NX]' if 'x' in chars else '[0.0, 0.0, 1]')
    lim_str += '\n    '
    lim_str += line.format('y',
                           '[-1.0, 1.0, NY]' if 'y' in chars else '[0.0, 0.0, 1]')
    lim_str += '\n    '
    lim_str += line.format('z',
                           '[-1.0, 1.0, NZ]' if 'z' in chars else '[0.0, 0.0, 1]')

    new = text.format(
        C1=C1,
        i1=['x', 'y', 'z'].index(c1),
        lim=lim_str,
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

    line = 'v{}lim = {}'
    lim_str = line.format(
        'x', '[-1.0, 1.0, NX]' if 'x' in chars else '[0.0, 0.0, 1]')
    lim_str += '\n    '
    lim_str += line.format('y',
                           '[-1.0, 1.0, NY]' if 'y' in chars else '[0.0, 0.0, 1]')
    lim_str += '\n    '
    lim_str += line.format('z',
                           '[-1.0, 1.0, NZ]' if 'z' in chars else '[0.0, 0.0, 1]')

    new = text.format(
        C1=C1,
        C2=C2,
        i1=['x', 'y', 'z'].index(c1),
        i2=['x', 'y', 'z'].index(c2),
        lim=lim_str,
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
