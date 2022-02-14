"""Plot error vs n_rebin."""
# pylint: disable=invalid-name

from . check_common import _plot_errors, _get_errors
from . init_layout import init_layout
from . ana import Parameters


def check_rebin_ipy(directories, names, custom_obs={}, Nmax0=100, ncols=3):
    """
    Plot error vs n_rebin in a Jupyter Widget.

    Parameters
    ----------
    directories : list of path-like objects
        Directories with bins to check.
    names : list of str
        Names of observables to check.
    Nmax0 : int, default=100
        Biggest n_rebin to consider. The default is 100.
    custom_obs : dict, default={}
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
    def __init__(self, directories, names, custom_obs={}, Nmax0=100, ncols=3):
        self.gui, self.log, self.axs, self.nrebin, self.select = \
            init_layout(directories, n_plots=len(names), ncols=ncols,
                        int_names=('N_rebin:',))
        self.nrebin.min = 1
        self.names = names
        self.custom_obs = custom_obs
        self.Nmax0 = Nmax0
        self.ncols = ncols

        self.init_dir()
        self.select.observe(self.update_select, 'value')
        self.nrebin.observe(self.update_nrebin, 'value')

    def init_dir(self):
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

    def update_select(self, change):
        del change
        with self.log:
            # display(change)
            self.init_dir()

    def update_nrebin(self, change):
        del change
        with self.log:
            if self.nrebin.value == self.par.N_rebin():
                return
            print(f'Change N_rebin to {self.nrebin.value}')
            self.par.set_N_rebin(self.nrebin.value)
            self.par.write_nml()
            for vert in self.verts:
                vert.set_xdata(self.nrebin.value)
