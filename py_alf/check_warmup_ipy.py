
import math

from . check_common import _get_bins, _replot
from . init_layout import init_layout
from . ana import Parameters


def check_warmup_ipy(directories, names, custom_obs={}, ncols=3):
    return CheckWarmupIpy(directories, names, custom_obs={}, ncols=3).gui


class CheckWarmupIpy:
    def __init__(self, directories, names, custom_obs={}, ncols=3):
        self.gui, self.log, self.axs, self.nmax, self.nskip, self.select = \
            init_layout(directories, n_plots=len(names), ncols=ncols,
                        int_names=('N_max:', 'N_skip:'))
        self.names = names
        self.custom_obs = custom_obs
        self.ncols = ncols
        
        self.init_dir()
        self.select.observe(self.update_select, 'value')
        self.nskip.observe(self.update_nskip, 'value')
        self.nmax.observe(self.update_nmax, 'value')
    
    def init_dir(self):
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
    
    
    def update_select(self, change):
        with self.log:
            # display(change)
            self.init_dir()
            
    def update_nskip(self, change):
        with self.log:
            if self.nskip.value == self.par.N_skip():
                return
            print(f'Change N_skip to {self.nskip.value}')
            self.par.set_N_skip(self.nskip.value)
            self.par.write_nml()
            for ax, name, bins in zip(self.axs, self.names, self.bins):
                _replot(ax, name, bins, self.par.N_skip())
            for ax in self.axs[-self.ncols:]:
                ax.set_xlabel('Bin number')
            
    def update_nmax(self, change):
        with self.log:
            self.axs[0].set_xlim(0.5, self.nmax.value+0.5)