#!/usr/bin/env python3
"""Plot bins to determine n_skip."""

import math
import tkinter as tk
import numpy as np

from scipy.optimize import curve_fit
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt

from alf_ana.ana import Parameters, ReadObs, read_scal


def check_warmup(directories, names, custom_obs={}):
    """
    Plot bins to determine n_skip. Opens a new window.

    Parameters
    ----------
    directories : list of path-like objects
        Directories with bins to check.
    names : list of str
        Names of observables to check.
    custom_obs : dict, default={}
        Defines additional observables derived from existing observables.
        See :func:`alf_ana.analysis.analysis`.
    """
    root = tk.Tk()

    n_dir_var = tk.IntVar(master=root, value=-1)
    directory_var = tk.StringVar(master=root)

    Nmax_str = tk.StringVar()
    N_skip_str = tk.StringVar()

    res = [None] * len(names)

    fig, axes = _create_fig(len(names))
    canvas = FigureCanvasTkAgg(fig, master=root)

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()

    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # def on_key_press(event):
    #     print("you pressed {}".format(event.key))
    #     key_press_handler(event, canvas, toolbar)

    # canvas.mpl_connect("key_press_event", on_key_press)

    def _set_nmax():
        Nmax = int(Nmax_str.get())
        axes[0].set_xlim(0.5, Nmax+0.5)
        canvas.draw()

    def _set_nskip():
        N_skip = int(N_skip_str.get())
        Nmax = int(Nmax_str.get())
        par = Parameters(directory_var.get())
        print("updating to N_skip={}".format(N_skip))
        par.set_N_skip(N_skip)
        par.write_nml()

        for n in range(len(res)):
            _replot(axes[n], names[n], res[n], N_skip, Nmax)
        canvas.draw()

    def _quit():
        root.quit()
        root.destroy()

    def _next():
        n_dir = n_dir_var.get() + 1
        if n_dir == len(directories):
            print("At end of list, click 'Finish' to exit.")
            return
        n_dir_var.set(n_dir)
        directory_var.set(directories[n_dir])
        root.wm_title('{} warmup'.format(directory_var.get()))
        par = Parameters(directory_var.get())
        _get_bins(directory_var.get(), names, custom_obs, res)

        for n, bins in enumerate(res):
            ax = axes[n]
            Nmax = bins.shape[0]
            _replot(ax, names[n], bins, par.N_skip(), Nmax)
        canvas.draw()
        Nmax_str.set(str(Nmax))
        N_skip_str.set(str(par.N_skip()))

    _next()

    frame = tk.Frame(root)
    frame.pack(side=tk.BOTTOM)

    nmax_frame = tk.Frame(frame)
    nmax_frame.pack(side=tk.LEFT)
    nmax_label = tk.Label(nmax_frame, text='N_max:')
    nmax_label.pack(side=tk.LEFT)
    nmax_entry = tk.Entry(nmax_frame, width=5, textvariable=Nmax_str)
    nmax_entry.pack()
    nmax_button = tk.Button(nmax_frame, text="Set", command=_set_nmax)
    nmax_button.pack(side=tk.RIGHT)

    nskip_frame = tk.Frame(frame)
    nskip_frame.pack(side=tk.LEFT)
    nskip_label = tk.Label(nskip_frame, text='N_skip:')
    nskip_label.pack(side=tk.LEFT)
    nskip_entry = tk.Entry(nskip_frame, width=5, textvariable=N_skip_str)
    nskip_entry.pack()
    nskip_button = tk.Button(nskip_frame, text="Set", command=_set_nskip)
    nskip_button.pack(side=tk.RIGHT)

    button_frame = tk.LabelFrame(frame, text='Quit')
    button_frame.pack(side=tk.RIGHT)
    button_next = tk.Button(button_frame, text="Next", command=_next)
    button_next.pack(side=tk.LEFT)
    button_quit = tk.Button(button_frame, text="Finish", command=_quit)
    button_quit.pack(side=tk.RIGHT)

    tk.mainloop()


def _create_fig(N):
    if N == 1:
        fig, axes0 = plt.subplots(1, 1, figsize=(10, 7), dpi=100)
        return fig, [axes0]
    ncols = math.ceil(math.sqrt(N))
    if ncols**2 - ncols >= N:
        nrows = ncols-1
    else:
        nrows = ncols

    fig, axes0 = plt.subplots(nrows, ncols, sharex='all',
                              figsize=(10, 7), dpi=100)
    return fig, axes0.flat


def _get_bins(directory, names, custom_obs, res):
    for n, obs_name in enumerate(names):
        if obs_name in custom_obs:
            print('custom', obs_name)
            obs_spec = custom_obs[obs_name]

            Bins = [ReadObs(directory, o, bare_bins=True)
                    for o in obs_spec['needs']]
            N_bins = Bins[0].N_bins

            bins = np.empty(N_bins)
            for i in range(N_bins):
                bins[i] = obs_spec['function'](
                    *[x for b in Bins for x in b.slice(i)],
                    **obs_spec['kwargs']).real
        else:
            print(obs_name)
            bins_c, sign, N_obs = read_scal(directory, obs_name,
                                            bare_bins=True)

            bins = bins_c[:, 0] / sign[:]
        res[n] = bins


def _replot(ax, o, bins, N_skip, Nmax):
    N_bins = len(bins)

    x = np.arange(1, N_bins+1)
    bins1 = bins[N_skip:]
    x1 = x[N_skip:]

    ax.clear()
    ax.set_title(o)
    ax.grid(True)
    ax.set_xlim(0.5, Nmax+0.5)

    ax.plot(range(1, N_bins+1), bins)
    ax.plot(x1, bins1, '.')

    m = np.mean(bins1)
    ax.plot([1, N_bins], [m, m])
    ax.plot([N_skip+1], [m], 'o')

    def func(x, y0, a):
        return y0 + a*x
    popt, pcov = curve_fit(func, x1, bins1)
    ax.plot(x1, func(x1, *popt))
    print(m, popt[1]/m)
