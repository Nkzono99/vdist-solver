import pickle as pkl
from .targets import VSolveTarget, BackTraceTraget


def save_vsolve_result(i: int, target: VSolveTarget, result):
    with open(target.data.directory / 'vtarget_{}.pkl'.format(i), 'wb') as f:
        pkl.dump(result, f)


def save_backtrace_result(i: int, target: BackTraceTraget, result):
    with open(target.data.directory / 'btarget_{}.pkl'.format(i), 'wb') as f:
        pkl.dump(result, f)


VSOLVE_AFTER = {
    'save': save_vsolve_result,
}

BACKTRACE_AFTER = {
    'save': save_backtrace_result,
}
