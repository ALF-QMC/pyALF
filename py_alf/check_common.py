"""Common resources for check_warmup and check_rebin."""
# pylint: disable=invalid-name
# pylint: disable=too-many-locals

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .ana import Parameters, ReadObs, custom_obs_get_dtype_len, error, jack, read_scal


def _create_fig(N):
    if N == 1:
        fig, axes0 = plt.subplots(1, 1, constrained_layout=True)
        return fig, [axes0]
    ncols = math.ceil(math.sqrt(N))
    nrows = ncols-1 if ncols**2 - ncols >= N else ncols

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

            dtype, length = custom_obs_get_dtype_len(obs_spec, Bins)
            del dtype
            bins = np.empty((N_bins, length))

            for i in range(N_bins):
                bins[i] = obs_spec['function'](
                    *[x for b in Bins for x in b.slice(i)],
                    **obs_spec['kwargs']).real
        else:
            print(obs_name)
            bins_c, sign, N_obs = read_scal(directory, obs_name,
                                            bare_bins=True)
            del N_obs

            bins = bins_c[:, 0].real / sign[:]
        res.append(bins)
    return res


def _replot(ax, obs_name, bins, N_skip, nmax=None):
    N_bins = len(bins)
    if nmax is None:
        nmax = N_bins

    x = np.arange(1, N_bins+1)
    bins1 = bins[N_skip:]
    x1 = x[N_skip:]

    ax.clear()
    ax.set_title(obs_name)
    ax.grid(True)
    ax.set_xlim(0.5, nmax+0.5)

    try:
        N_obs = bins1.shape[1]
    except IndexError:
        N_obs = 1
    for i in range(N_obs):
        bins2 = bins1.reshape((len(x1),)) if N_obs == 1 else bins1[:, i]

        p = ax.plot(range(1, N_bins+1), bins)
        color = None if N_obs == 1 else p[0].get_color()
        ax.plot(x1, bins2, '.', c=color)

        m = np.mean(bins2)
        ax.plot([1, N_bins], [m, m], c=color, ls='-.')
        ax.plot([N_skip+1], [m], 'o', c="red")

        def func(x, y0, a):
            return y0 + a*x
        popt, pcov, *_ = curve_fit(func, x1, bins2)
        del pcov
        ax.plot(x1, func(x1, *popt), c=color)
        print(m, popt[1]/m)
    ax.axvline(x=N_skip+0.5, color="red")


def _rebin_err(bins, N_skip, Nmax):
    N_obs = bins.shape[1]
    res = np.empty((Nmax, N_obs, 2))
    for N in range(1, Nmax+1):
        J = jack(bins, par=None, N_skip=N_skip, N_rebin=N)
        m, e = error(J[:, :])  # pylint: disable=unbalanced-tuple-unpacking
        res[N-1, :, 0] = m
        res[N-1, :, 1] = e
    return res


def _plot_errors(axs, errs, obs_names, custom_obs):
    for ax, err, obs_name in zip(axs, errs, obs_names):
        ax.clear()
        ax.grid(True)
        ax.set_title(f'{obs_name}_err')
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
            dtype, length = custom_obs_get_dtype_len(obs_spec, bins)
            err = np.empty((Nmax, length))

            for N_rebin in range(1, Nmax+1):
                jacks = [x for b in bins for x in b.jack(N_rebin)]

                N_bins = len(jacks[0])
                J = np.empty((N_bins, length), dtype=dtype)
                print(f'{N_rebin}*{N_bins}={N_rebin*N_bins}')
                for i in range(N_bins):
                    J[i] = obs_spec['function'](*[x[i] for x in jacks],
                                                **obs_spec['kwargs'])
                m, e = error(J)  # pylint: disable=unbalanced-tuple-unpacking
                del m
                err[N_rebin-1] = e
        elif obs_name.endswith('_scal'):
            print(obs_name)
            bins_c, sign, N_obs = read_scal(directory, obs_name,
                                            bare_bins=True)
            del sign, N_obs
            N_bins = bins_c.shape[0] - N_skip
            Nmax = min(N_bins // 3, Nmax0)
            err = _rebin_err(bins_c, N_skip, Nmax)
        else:
            raise TypeError(f'Illegal observable {obs_name}')
        # print(err)
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
