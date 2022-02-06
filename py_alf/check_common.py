"""Common resources for check_warmup and check_rebin."""
# pylint: disable=invalid-name

import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from . ana import Parameters, ReadObs, read_scal, jack, error


def _create_fig(N):
    if N == 1:
        fig, axes0 = plt.subplots(1, 1, constrained_layout=True)
        return fig, [axes0]
    ncols = math.ceil(math.sqrt(N))
    if ncols**2 - ncols >= N:
        nrows = ncols-1
    else:
        nrows = ncols

    fig, axes0 = plt.subplots(
        nrows, ncols,
        sharex='all',
        constrained_layout=True,
    )
    return fig, list(axes0.flat)


def _get_bins(directory, names, custom_obs):
    res = []
    for obs_name in names:
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
        res.append(bins)
    return res


def _replot(ax, obs_name, bins, N_skip):
    N_bins = len(bins)

    x = np.arange(1, N_bins+1)
    bins1 = bins[N_skip:]
    x1 = x[N_skip:]

    ax.clear()
    ax.set_title(obs_name)
    ax.grid(True)
    ax.set_xlim(0.5, len(bins)+0.5)

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


def _rebin_err(bins, N_skip, Nmax):
    N_obs = bins.shape[1]
    res = np.empty((Nmax, N_obs, 2))
    for N in range(1, Nmax+1):
        J = jack(bins, par=None, N_skip=N_skip, N_rebin=N)
        m, e = error(J[:, :])
        res[N-1, :, 0] = m
        res[N-1, :, 1] = e
    return res


def _plot_errors(axs, errs, obs_names, custom_obs):
    for ax, err, obs_name in zip(axs, errs, obs_names):
        ax.clear()
        ax.grid(True)
        ax.set_title('{}_err'.format(obs_name))
        if obs_name in custom_obs:
            ax.plot(range(1, len(err)+1), err)
            ax.set_ylim(err.min(), err.max())
        else:
            for i in range(err.shape[1]):
                ax.plot(range(1, len(err)+1), err[:, i, 1])
            ax.set_ylim(err[:, :, 1].min(), err[:, :, 1].max())


def _get_errors(directory, names, custom_obs, Nmax0):
    res = []
    N_skip = Parameters(directory).N_skip()
    for obs_name in names:
        if obs_name in custom_obs:
            print('custom', obs_name)
            obs_spec = custom_obs[obs_name]
            bins = [ReadObs(directory, o, bare_bins=True)
                    for o in obs_spec['needs']]

            N_bins1 = bins[0].N_bins - N_skip
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
        elif obs_name.endswith('_scal'):
            print(obs_name)
            bins_c, sign, N_obs = read_scal(directory, obs_name,
                                            bare_bins=True)
            N_bins = bins_c.shape[0] - N_skip
            Nmax = min(N_bins // 3, Nmax0)
            err = _rebin_err(bins_c, N_skip, Nmax)
        else:
            raise Exception(f'Illegal observable {obs_name}')
        print(err)
        res.append(err)
    return res


# def _auto_corr(bins, Nmax):
#     N_bins = len(bins)
#     if N_bins < Nmax:
#         raise Exception("Number of bins too low")

#     res = np.empty((Nmax,) + bins.shape[1:], dtype=bins.dtype)

#     for n in range(1, Nmax+1):
#         X1 = np.zeros(bins.shape[1:], dtype=bins.dtype)
#         X2 = np.zeros(bins.shape[1:], dtype=bins.dtype)
#         X3 = np.zeros(bins.shape[1:], dtype=bins.dtype)
#         for i in range(N_bins-n):
#             X3 += bins[i]
#         X3 /= (N_bins-n)

#         for i in range(N_bins-n):
#             X1 += (bins[i]-X3) * (bins[i+n]-X3)
#             X2 += (bins[i]-X3) * (bins[i]-X3)

#         res[n-1] = X1/X2
#     return res
