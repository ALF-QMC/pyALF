"""Plot error vs n_rebin."""
# pylint: disable=invalid-name

from .ana import Parameters
from .check_common import _get_errors, _plot_errors
from .init_layout import init_layout


def check_rebin_ipy(directories, names, custom_obs=None, Nmax0=100, ncols=3):
    """Plot error vs n_rebin in a Jupyter Widget.

    Parameters
    ----------
    directories : list of path-like objects
        Directories with bins to check.
    names : list of str
        Names of observables to check.
    Nmax0 : int, default=100
        Biggest n_rebin to consider. The default is 100.
    custom_obs : dict, default=None
        Defines additional observables derived from existing observables.
        See :func:`py_alf.analysis`.

    Returns
    -------
    Jupyter Widget
        A graphical user interface based on ipywidgets

    """
    return CheckRebinIpy(
        directories, names, custom_obs=custom_obs, Nmax0=Nmax0, ncols=ncols).gui


class CheckRebinIpy:
    """Plot error vs n_rebin in a Jupyter Widget.

    Parameters
    ----------
    directories : list of path-like objects
        Directories with bins to check.
    names : list of str
        Names of observables to check.
    Nmax0 : int, default=100
        Biggest n_rebin to consider. The default is 100.
    custom_obs : dict, default=None
        Defines additional observables derived from existing observables.
        See :func:`py_alf.analysis`.

    Returns
    -------
    Jupyter Widget
        A graphical user interface based on ipywidgets

    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=too-few-public-methods
    def __init__(self, directories, names, custom_obs=None, Nmax0=100, ncols=3):
        self.gui, self.log, self.axs, self.nrebin, self.select = \
            init_layout(directories, n_plots=len(names), ncols=ncols,
                        int_names=('N_rebin:',))
        self.nrebin.min = 1
        self.names = names
        if custom_obs is None:
            self.custom_obs = {}
        else:
            self.custom_obs = custom_obs
        self.Nmax0 = Nmax0
        self.ncols = ncols

        self._init_dir()
        self.select.observe(self._update_select, 'value')
        self.nrebin.observe(self._update_nrebin, 'value')

    def _init_dir(self):
        with self.log:
            self.par = Parameters(self.select.value)
            errors = _get_errors(
                self.select.value, self.names, self.custom_obs, self.Nmax0)
            _plot_errors(self.axs, errors, self.names, self.custom_obs)
            self.nrebin.max = len(errors[0])
            self.nrebin.value = self.par.N_rebin()

            self.verts = []
            for ax in self.axs:
                self.verts.append(ax.axvline(x=self.nrebin.value, color="red"))
            for ax in self.axs[-self.ncols:]:
                ax.set_xlabel('N_rebin')

    def _update_select(self, change):
        del change
        with self.log:
            # display(change)
            self._init_dir()

    def _update_nrebin(self, change):
        del change
        with self.log:
            if self.nrebin.value == self.par.N_rebin():
                return
            print(f'Change N_rebin to {self.nrebin.value}')
            self.par.set_N_rebin(self.nrebin.value)
            self.par.write_nml()
            for vert in self.verts:
                vert.set_xdata([self.nrebin.value])
