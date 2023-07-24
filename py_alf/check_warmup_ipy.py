"""Plot bins to determine n_skip."""
# pylint: disable=invalid-name

import math

from . check_common import _get_bins, _replot
from . init_layout import init_layout
from . ana import Parameters


def check_warmup_ipy(directories, names, custom_obs=None, ncols=3):
    """
    Plot bins to determine n_skip in a Jupyter Widget.

    Parameters
    ----------
    directories : list of path-like objects
        Directories with bins to check.
    names : list of str
        Names of observables to check.
    custom_obs : dict, default=None
        Defines additional observables derived from existing observables.
        See :func:`py_alf.analysis`.

    Returns
    -------
    Jupyter Widget
        A graphical user interface based on ipywidgets
    """
    return CheckWarmupIpy(
        directories, names, custom_obs=custom_obs, ncols=ncols).gui


class CheckWarmupIpy:  # pylint: disable=too-few-public-methods
    """
    Plot bins to determine n_skip in a Jupyter Widget.

    Parameters
    ----------
    directories : list of path-like objects
        Directories with bins to check.
    names : list of str
        Names of observables to check.
    custom_obs : dict, default=None
        Defines additional observables derived from existing observables.
        See :func:`py_alf.analysis`.

    Returns
    -------
    Jupyter Widget
        A graphical user interface based on ipywidgets
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self, directories, names, custom_obs=None, ncols=3):
        self.gui, self.log, self.axs, self.nmax, self.nskip, self.select = \
            init_layout(directories, n_plots=len(names), ncols=ncols,
                        int_names=('N_max:', 'N_skip:'))
        self.names = names
        if custom_obs is None:
            custom_obs = {}
        else:
            self.custom_obs = custom_obs
        self.ncols = ncols

        self._init_dir()
        self.select.observe(self._update_select, 'value')
        self.nskip.observe(self._update_nskip, 'value')
        self.nmax.observe(self._update_nmax, 'value')

    def _init_dir(self):
        """Initial setup of data from currently selected directory."""
        with self.log:
            self.bins = _get_bins(
                self.select.value, self.names, self.custom_obs)
            self.par = Parameters(self.select.value)

            nmax = math.inf
            for ax, name, bins in zip(self.axs, self.names, self.bins):
                nmax = min(nmax, len(bins))
                _replot(ax, name, bins, self.par.N_skip())
            for ax in self.axs[-self.ncols:]:
                ax.set_xlabel('Bin number')
            self.nskip.max = nmax+2
            self.nskip.value = self.par.N_skip()
            self.nmax.max = nmax
            self.nmax.value = nmax

    def _update_select(self, change):
        del change
        with self.log:
            # display(change)
            self._init_dir()

    def _update_nskip(self, change):
        del change
        with self.log:
            if self.nskip.value == self.par.N_skip():
                return
            print(f'Change N_skip to {self.nskip.value}')
            self.par.set_N_skip(self.nskip.value)
            self.par.write_nml()
            for ax, name, bins in zip(self.axs, self.names, self.bins):
                _replot(ax, name, bins, self.par.N_skip(), self.nmax.value)
            for ax in self.axs[-self.ncols:]:
                ax.set_xlabel('Bin number')

    def _update_nmax(self, change):
        del change
        with self.log:
            self.axs[0].set_xlim(0.5, self.nmax.value+0.5)
