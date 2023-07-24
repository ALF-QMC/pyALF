"""Plot error vs n_rebin."""
# pylint: disable=invalid-name

import tkinter as tk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from . ana import Parameters
from . check_common import _create_fig, _plot_errors, _get_errors


class check_rebin_tk():
    """
    Plot error vs n_rebin. Opens a new window.

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
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-few-public-methods

    def __init__(self, directories, names, Nmax0=100, custom_obs=None):
        self.directories = directories
        self.names = names
        self.Nmax0 = Nmax0
        if custom_obs is None:
            custom_obs = {}
        else:
            self.custom_obs = custom_obs
        self.root = tk.Tk()

        self.n_dir_var = tk.IntVar(master=self.root, value=-1)
        self.directory_var = tk.StringVar(master=self.root)

        self.N_rebin_str = tk.StringVar()

        self.fig, self.axs = _create_fig(len(self.names))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)

        toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        toolbar.update()

        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self._next()

        frame = tk.Frame(self.root)
        frame.pack(side=tk.BOTTOM)

        nskip_frame = tk.Frame(frame)
        nskip_frame.pack(side=tk.LEFT)
        nskip_label = tk.Label(nskip_frame, text='N_rebin:')
        nskip_label.pack(side=tk.LEFT)
        nskip_entry = tk.Entry(
            nskip_frame, width=5, textvariable=self.N_rebin_str)
        nskip_entry.pack()
        nskip_button = tk.Button(
            nskip_frame, text="Set", command=self._set_nrebin)
        nskip_button.pack(side=tk.RIGHT)

        button_frame = tk.LabelFrame(frame, text='Quit')
        button_frame.pack(side=tk.RIGHT)
        button_next = tk.Button(
            button_frame, text="Next", command=self._next)
        button_next.pack(side=tk.LEFT)
        button_quit = tk.Button(
            button_frame, text="Finish", command=self._quit)
        button_quit.pack(side=tk.RIGHT)

        tk.mainloop()

    def _set_nrebin(self):
        N_rebin = int(self.N_rebin_str.get())
        print("updating to N_rebin={}".format(N_rebin))
        self.par.set_N_rebin(N_rebin)
        self.par.write_nml()
        for vert in self.verts:
            vert.set_xdata(self.par.N_rebin())
        self.canvas.draw()

    def _quit(self):
        self.root.quit()
        self.root.destroy()

    def _next(self):
        n_dir = self.n_dir_var.get() + 1
        if n_dir == len(self.directories):
            print("At end of list. Click 'Finish' to exit.")
            return
        self.n_dir_var.set(n_dir)
        self.directory_var.set(self.directories[n_dir])
        self.root.wm_title(
            '{} N_rebin vs error'.format(self.directory_var.get()))
        self.par = Parameters(self.directory_var.get())

        _plot_errors(
            self.axs,
            _get_errors(self.directory_var.get(), self.names,
                        self.custom_obs, self.Nmax0),
            self.names, self.custom_obs
        )
        self.verts = []
        for ax in self.axs:
            self.verts.append(ax.axvline(x=self.par.N_rebin(), color="red"))
        self.canvas.draw()
        self.N_rebin_str.set(str(self.par.N_rebin()))
