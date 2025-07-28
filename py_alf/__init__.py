"""pyALF, a Python package for the Algorithms for Lattice Fermions (ALF)."""
# pylint: disable=inconsistent-return-statements
# pylint: disable=import-outside-toplevel

# Classes
from .alf_source import ALF_source
from .lattice import Lattice
from .simulation import Simulation

__all__ = ['ALF_source', 'Simulation', 'Lattice']


def check_warmup(*args, gui='tk', **kwargs):
    """Plot bins to determine n_skip.

    Calls either :func:`py_alf.check_warmup_tk` or
    :func:`py_alf.check_warmup_ipy`.

    Parameters
    ----------
    *args
    gui : {"tk", "ipy"}
    **kwargs

    """
    if gui == 'tk':
        from .check_warmup_tk import check_warmup_tk
        check_warmup_tk(*args, **kwargs)
    elif gui == 'ipy':
        from .check_warmup_ipy import check_warmup_ipy
        return check_warmup_ipy(*args, **kwargs)
    else:
        raise TypeError(f'Illegal value gui={gui}')

def check_rebin(*args, gui='tk', **kwargs):
    """Plot error vs n_rebin in a Jupyter Widget.

    Calls either :func:`py_alf.check_rebin_tk` or
    :func:`py_alf.check_rebin_ipy`.

    Parameters
    ----------
    *args
    gui : {"tk", "ipy"}
    **kwargs

    """
    if gui == 'tk':
        from .check_rebin_tk import check_rebin_tk
        check_rebin_tk(*args, **kwargs)
    elif gui == 'ipy':
        from .check_rebin_ipy import check_rebin_ipy
        return check_rebin_ipy(*args, **kwargs)
    else:
        raise TypeError(f'Illegal value gui={gui}')
