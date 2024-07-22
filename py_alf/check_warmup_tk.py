"""Plot bins to determine n_skip."""
# pylint: disable=invalid-name

import tkinter as tk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .ana import Parameters
from .check_common import _create_fig, _get_bins, _replot


class check_warmup_tk:  # pylint: disable=too-few-public-methods
    """Plot bins to determine n_skip. Opens a new window.

    Parameters
    ----------
    directories : list of path-like objects
        Directories with bins to check.
    names : list of str
        Names of observables to check.
    custom_obs : dict, default=None
        Defines additional observables derived from existing observables.
        See :func:`py_alf.analysis`.

    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-locals

    def __init__(self, directories, names, custom_obs=None):
        self.directories = directories
        self.names = names
        if custom_obs is None:
            self.custom_obs = {}
        else:
            self.custom_obs = custom_obs
        self.root = tk.Tk()

        self.n_dir_var = tk.IntVar(master=self.root, value=-1)
        self.directory_var = tk.StringVar(master=self.root)

        self.Nmax_str = tk.StringVar()
        self.N_skip_str = tk.StringVar()

        self.fig, self.axes = _create_fig(len(names))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()

        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self._next()

        frame = tk.Frame(self.root)
        frame.pack(side=tk.BOTTOM)

        nmax_frame = tk.Frame(frame)
        nmax_frame.pack(side=tk.LEFT)
        nmax_label = tk.Label(nmax_frame, text='N_max:')
        nmax_label.pack(side=tk.LEFT)
        nmax_entry = tk.Entry(
            nmax_frame, width=5, textvariable=self.Nmax_str)
        nmax_entry.pack()
        nmax_button = tk.Button(
            nmax_frame, text="Set", command=self._set_nmax)
        nmax_button.pack(side=tk.RIGHT)

        nskip_frame = tk.Frame(frame)
        nskip_frame.pack(side=tk.LEFT)
        nskip_label = tk.Label(nskip_frame, text='N_skip:')
        nskip_label.pack(side=tk.LEFT)
        nskip_entry = tk.Entry(
            nskip_frame, width=5, textvariable=self.N_skip_str)
        nskip_entry.pack()
        nskip_button = tk.Button(
            nskip_frame, text="Set", command=self._set_nskip)
        nskip_button.pack(side=tk.RIGHT)

        button_frame = tk.LabelFrame(frame, text='Quit')
        button_frame.pack(side=tk.RIGHT)
        button_next = tk.Button(
            button_frame, text="Next", command=self._next)
        button_next.pack(side=tk.LEFT)
        button_quit = tk.Button(
            button_frame, text="Finish",command=self._quit)
        button_quit.pack(side=tk.RIGHT)
        tk.mainloop()

    def _set_nmax(self):
        self.axes[0].set_xlim(0.5, int(self.Nmax_str.get())+0.5)
        self.canvas.draw()

    def _set_nskip(self):
        N_skip = int(self.N_skip_str.get())
        print(f"updating to N_skip={N_skip}")
        self.par.set_N_skip(N_skip)
        self.par.write_nml()

        for ax, name, bins in zip(self.axes, self.names, self.res):
            _replot(ax, name, bins, N_skip)
        self.canvas.draw()

    def _quit(self):
        self.root.quit()
        self.root.destroy()

    def _next(self):
        n_dir = self.n_dir_var.get() + 1
        if n_dir == len(self.directories):
            print("At end of list, click 'Finish' to exit.")
            return
        self.n_dir_var.set(n_dir)
        self.directory_var.set(self.directories[n_dir])
        self.root.wm_title(f'{self.directory_var.get()} warmup')
        self.par = Parameters(self.directory_var.get())
        self.res = _get_bins(
            self.directory_var.get(), self.names, self.custom_obs)

        Nmax = np.inf
        for ax, name, bins in zip(self.axes, self.names, self.res):
            Nmax = min(Nmax, len(bins))
            _replot(ax, name, bins, self.par.N_skip())
        self.canvas.draw()
        self.Nmax_str.set(str(Nmax))
        self.N_skip_str.set(str(self.par.N_skip()))
