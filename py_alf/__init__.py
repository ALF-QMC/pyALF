"""
pyALF, a Python package for the Algorithms for Lattice Fermions (ALF).
"""
# pylint: disable=inconsistent-return-statements

# Module containing low-level analysis functions
from . import ana

# Module containing utility function
from . import utils

# Classes
from . alf_source import ALF_source
from . simulation import Simulation
from . lattice import Lattice

# High-level analysis functions
from . analysis import analysis
## With Tkinter based GUI
from . check_warmup_tk import check_warmup_tk
from . check_rebin_tk import check_rebin_tk
## Using Jupyter Widgets
from . check_warmup_ipy import check_warmup_ipy
from . check_rebin_ipy import check_rebin_ipy

def check_warmup(*args, gui='tk', **kwargs):
    """
    Plot bins to determine n_skip.

    Calls either :func:`py_alf.check_warmup_tk` or
    :func:`py_alf.check_warmup_ipy`. 

    Parameters
    ----------
    *args
    gui : {"tk", "ipy"}
    **kwargs
    """
    if gui == 'tk':
        check_warmup_tk(*args, **kwargs)
    elif gui == 'ipy':
        return check_warmup_ipy(*args, **kwargs)
    else:
        raise TypeError(f'Illegal value gui={gui}')

def check_rebin(*args, gui='tk', **kwargs):
    """
    Plot error vs n_rebin in a Jupyter Widget.

    Calls either :func:`py_alf.check_rebin_tk` or
    :func:`py_alf.check_rebin_ipy`. 

    Parameters
    ----------
    *args
    gui : {"tk", "ipy"}
    **kwargs
    """
    if gui == 'tk':
        check_rebin_tk(*args, **kwargs)
    elif gui == 'ipy':
        return check_rebin_ipy(*args, **kwargs)
    else:
        raise TypeError(f'Illegal value gui={gui}')
