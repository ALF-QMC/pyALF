#!/usr/bin/env python3
"""Plot error vs n_rebin"""

import math
import tkinter as tk

import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt


from alf_ana.ana import Parameters, jack, error, read_scal, ReadObs


def check_rebin(directories, names, Nmax0=100, custom_obs={}):
    root = tk.Tk()

    n_dir_var = tk.IntVar(master=root, value=-1)
    directory_var = tk.StringVar(master=root)

    # Nmax_str = tk.StringVar()
    N_rebin_str = tk.StringVar()

    fig, axes = create_fig(len(names))
    vert = [None] * len(names)
    canvas = FigureCanvasTkAgg(fig, master=root)

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()

    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def on_key_press(event):
        print("you pressed {}".format(event.key))
        key_press_handler(event, canvas, toolbar)

    canvas.mpl_connect("key_press_event", on_key_press)

    def _set_nrebin():
        N_rebin = int(N_rebin_str.get())
        par = Parameters(directory_var.get())
        print("updating to N_rebin={}".format(N_rebin))
        par.set_N_rebin(N_rebin)
        par.write_nml()
        for i in vert:
            i.set_xdata(par.N_rebin())
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
        root.wm_title('{} N_rebin vs error'.format(directory_var.get()))
        par = Parameters(directory_var.get())

        for n, obs_name in enumerate(names):
            ax = axes[n]
            ax.clear()
            ax.grid(True)
            ax.set_title(obs_name + '_err')
            vert[n] = ax.axvline(x=par.N_rebin(), color="red")
            if obs_name in custom_obs:
                print('custom', obs_name)
                obs_spec = custom_obs[obs_name]
                bins = [ReadObs(directory_var.get(), o, bare_bins=True)
                        for o in obs_spec['needs']]

                N_bins1 = bins[0].N_bins - par.N_skip()
                Nmax = min(N_bins1 // 3, Nmax0)
                err = np.empty(Nmax)

                for N in range(1, Nmax+1):
                    jacks = [x for b in bins for x in b.jack(N)]

                    N_bins = len(jacks[0])
                    J = np.empty(N_bins, dtype=jacks[0].dtype)
                    print(N_bins)
                    for i in range(N_bins):
                        J[i] = obs_spec['function'](*[x[i] for x in jacks],
                                                    **obs_spec['kwargs'])

                    m, e = error(J)
                    err[N-1] = e
                print(err)

                ax.plot(range(1, Nmax+1), err)
                ax.set_ylim(err.min(), err.max())

            else:
                print(obs_name)
                bins_c, sign, N_obs = read_scal(directory_var.get(), obs_name,
                                                bare_bins=True)
                N_bins = bins_c.shape[0] - par.N_skip()
                Nmax = min(N_bins // 3, Nmax0)
                err = rebin_err(bins_c, par, Nmax)
                ax.set_title('{}_err'.format(obs_name))
                for i in range(err.shape[1]):
                    ax.plot(range(1, Nmax+1), err[:, i, 1])

                    with open('{}_{}_err'.format(obs_name, i), 'w') as f2:
                        for l in range(Nmax):
                            f2.write('{} {} {}\n'.format(
                                l+1, err[l, i, 0], err[l, i, 1]))
                ax.set_ylim(err[:, :, 1].min(), err[:, :, 1].max())
        canvas.draw()
        N_rebin_str.set(str(par.N_rebin()))

    _next()

    frame = tk.Frame(root)
    frame.pack(side=tk.BOTTOM)

    nskip_frame = tk.Frame(frame)
    nskip_frame.pack(side=tk.LEFT)
    nskip_label = tk.Label(nskip_frame, text='N_rebin:')
    nskip_label.pack(side=tk.LEFT)
    nskip_entry = tk.Entry(nskip_frame, width=5, textvariable=N_rebin_str)
    nskip_entry.pack()
    nskip_button = tk.Button(nskip_frame, text="Set", command=_set_nrebin)
    nskip_button.pack(side=tk.RIGHT)

    button_frame = tk.LabelFrame(frame, text='Quit')
    button_frame.pack(side=tk.RIGHT)
    button_next = tk.Button(button_frame, text="Next", command=_next)
    button_next.pack(side=tk.LEFT)
    button_quit = tk.Button(button_frame, text="Finish", command=_quit)
    button_quit.pack(side=tk.RIGHT)

    tk.mainloop()

def create_fig(N):
    if N == 1:
        return plt.subplots(1, 1, figsize=(10, 7), dpi=100)
    ncols = math.ceil(math.sqrt(N))
    if ncols**2 - ncols >= N:
        nrows = ncols-1
    else:
        nrows = ncols

    fig, axes0 = plt.subplots(nrows, ncols, sharex='all',
                              figsize=(10, 7), dpi=100)
    return fig, axes0.flat


def auto_corr(bins, Nmax):
    N_bins = len(bins)
    if N_bins < Nmax:
        raise Exception("Number of bins too low")

    res = np.empty((Nmax,) + bins.shape[1:], dtype=bins.dtype)

    for n in range(1, Nmax+1):
        X1 = np.zeros(bins.shape[1:], dtype=bins.dtype)
        X2 = np.zeros(bins.shape[1:], dtype=bins.dtype)
        X3 = np.zeros(bins.shape[1:], dtype=bins.dtype)
        for i in range(N_bins-n):
            X3 += bins[i]
        X3 /= (N_bins-n)

        for i in range(N_bins-n):
            X1 += (bins[i]-X3) * (bins[i+n]-X3)
            X2 += (bins[i]-X3) * (bins[i]-X3)

        res[n-1] = X1/X2
    return res


def rebin_err(bins, par, Nmax):
    N_obs = bins.shape[1]
    res = np.empty((Nmax, N_obs, 2))

    for N in range(1, Nmax+1):
        J = jack(bins, par, N_rebin=N)
        m, e = error(J[:, :])
        res[N-1, :, 0] = m
        res[N-1, :, 1] = e

    return res
