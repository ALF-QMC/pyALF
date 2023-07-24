"""Supplies the default analysis routine."""
# pylint: disable=invalid-name

import os
import pickle

import h5py
import numpy as np

from . ana import (Parameters, ReadObs, error, ana_scal, ana_hist,
                   ana_eq, write_res_eq, ana_tau, write_res_tau,
                   custom_obs_get_dtype_len)
from . exceptions import TooFewBinsError


def analysis(directory,
             symmetry=None, custom_obs=None, do_tau=True, always=False):
    """
    Perform analysis in the given directory.

    Results are written to the pickled dictionary `res.pkl` and in plain text
    in the folder `res/`.

    Parameters
    ----------
    directory : path-like object
        Directory containing Monte Carlo bins.
    symmetry : list of functions, optional
        List of functions reppresenting symmetry operations on lattice,
        including unity. It is used to symmetrize lattice-type
        observables.
    custom_obs : dict, default=None
        Defines additional observables derived from existing observables.
        The key of each entry is the observable name and the value is a
        dictionary with the format::

            {'needs': some_list,
             'kwargs': some_dict,
             'function': some_function,}

        `some_list` contains observable names to be read by
        :class:`py_alf.ana.ReadObs`. Jackknife bins and kwargs from
        `some_dict` are handed to `some_function` with a separate call for
        each bin.
    do_tau : bool, default=True
        Analyze time-displaced correlation functions. Setting this to False
        speeds up analysis and makes result files much smaller.
    always : bool, default=False
        Do not skip if parameters and bins are older than results.
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    print('### Analyzing {} ###'.format(directory))
    print(os.getcwd())

    par = Parameters(directory)
    if 'data.h5' in os.listdir(directory):
        if not always:
            try:
                d1 = os.path.getmtime(os.path.join(directory, 'data.h5')) \
                    - os.path.getmtime(os.path.join(directory, 'res.pkl'))
                d2 = os.path.getmtime(os.path.join(directory, 'parameters')) \
                    - os.path.getmtime(os.path.join(directory, 'res.pkl'))
                if d1 < 0 and d2 < 0:
                    print('already analyzed')
                    return
            except OSError:
                pass

        # pylint: disable=no-member
        with h5py.File(os.path.join(directory, 'data.h5'), "r") as f:
            params = {}
            for name in f['parameters']:
                params.update(f['parameters'][name].attrs)
            list_obs = []
            list_scal = []
            list_hist = []
            list_eq = []
            list_tau = []

            N_bins = 0
            for o in f:
                if o.endswith('_scal'):
                    list_obs.append(o)
                    list_scal.append(o)
                    N_bins = max([N_bins, f[o + "/sign"].shape[0]])
                elif o.endswith('_hist'):
                    list_obs.append(o)
                    list_hist.append(o)
                    N_bins = max([N_bins, f[o + "/sign"].shape[0]])
                elif o.endswith('_eq'):
                    list_obs.append(o)
                    list_eq.append(o)
                    N_bins = max([N_bins, f[o + "/sign"].shape[0]])
                elif o.endswith('_tau'):
                    list_obs.append(o)
                    list_tau.append(o)
                    N_bins = max([N_bins, f[o + "/sign"].shape[0]])

        if N_bins < par.N_min():
            print('too few bins ', N_bins)
            return
    else:
        params = {}  # Stays empty, parameters are only supported with HDF5
        list_obs = []
        list_scal = []
        list_hist = []
        list_eq = []
        list_tau = []
        for o in os.listdir(directory):
            if o.endswith('_scal'):
                list_obs.append(o)
                list_scal.append(o)
            elif o.endswith('_hist'):
                list_obs.append(o)
                list_hist.append(o)
            elif o.endswith('_eq'):
                list_obs.append(o)
                list_eq.append(o)
            elif o.endswith('_tau'):
                list_obs.append(o)
                list_tau.append(o)

    if 'res' not in os.listdir(directory):
        os.mkdir(os.path.join(directory, 'res'))

    dic = params

    if custom_obs is not None:
        print("Custom observables:")
        for obs_name, obs_spec in custom_obs.items():
            if all(x in list_obs for x in obs_spec['needs']):
                print('custom', obs_name, obs_spec['needs'])
                jacks = [ReadObs(directory, obs_name)
                         for obs_name in obs_spec['needs']]

                N_bins = jacks[0].N_bins
                dtype, length = custom_obs_get_dtype_len(obs_spec, jacks)
                if length == 1:
                    shape = (N_bins,)
                else:
                    shape = (N_bins, length)
                J = np.empty(shape, dtype=dtype)
                for i in range(N_bins):
                    J[i] = obs_spec['function'](
                        *[x for j in jacks for x in j.slice(i)],
                        **obs_spec['kwargs'])

                dat = error(J)

                dic[obs_name] = dat[0]
                dic[obs_name+'_err'] = dat[1]

                np.savetxt(
                    os.path.join(directory, 'res', obs_name),
                    dat
                )

    print("Scalar observables:")
    for obs_name in list_scal:
        print(obs_name)
        try:
            sign, dat = ana_scal(directory, obs_name)
        except TooFewBinsError:
            print("Too few bins, skipping.")
            continue

        dic[obs_name+'_sign'] = sign[0]
        dic[obs_name+'_sign_err'] = sign[1]
        for i in range(len(dat)):
            dic[obs_name+str(i)] = dat[i, 0]
            dic[obs_name+str(i)+'_err'] = dat[i, 1]

        np.savetxt(
            os.path.join(directory, 'res', obs_name),
            dat,
            header='Sign: {} {}'.format(*sign)
        )

    print("Histogram observables:")
    for obs_name in list_hist:
        print(obs_name)
        try:
            sign, above, below, dat, upper, lower = ana_hist(directory, obs_name)
        except TooFewBinsError:
            print("Too few bins, skipping.")
            continue

        hist = {}
        hist['dat'] = dat
        hist['sign'] = sign
        hist['above'] = above
        hist['below'] = below
        hist['upper'] = upper
        hist['lower'] = lower
        dic[obs_name] = hist

        np.savetxt(
            os.path.join(directory, 'res', obs_name),
            dat,
            header='Sign: {} {}, above {} {}, below {} {}'.format(
                *sign, *above, *below)
        )

    print("Equal time observables:")
    for obs_name in list_eq:
        print(obs_name)
        try:
            sign, m_k, e_k, m_k_sum, e_k_sum, m_r, e_r, m_r_sum, e_r_sum, latt = \
                ana_eq(directory, obs_name, sym=symmetry)
        except TooFewBinsError:
            print("Too few bins, skipping.")
            continue

        write_res_eq(directory, obs_name,
                     m_k, e_k, m_k_sum, e_k_sum,
                     m_r, e_r, m_r_sum, e_r_sum, latt)

        dic[obs_name+'K'] = m_k
        dic[obs_name+'K_err'] = e_k
        dic[obs_name+'K_sum'] = m_k_sum
        dic[obs_name+'K_sum_err'] = e_k_sum
        dic[obs_name+'R'] = m_r
        dic[obs_name+'R_err'] = e_r
        dic[obs_name+'R_sum'] = m_r_sum
        dic[obs_name+'R_sum_err'] = e_r_sum
        dic[obs_name+'_lattice'] = {
            'L1': latt.L1,
            'L2': latt.L2,
            'a1': latt.a1,
            'a2': latt.a2
        }

    if do_tau:
        print("Time displaced observables:")
        for obs_name in list_tau:
            print(obs_name)
            try:
                sign, m_k, e_k, m_r, e_r, dtau, latt = \
                    ana_tau(directory, obs_name, sym=symmetry)
            except TooFewBinsError:
                print("Too few bins, skipping.")
                continue

            write_res_tau(directory, obs_name,
                          m_k, e_k, m_r, e_r, dtau, latt)

            dic[obs_name+'K'] = m_k
            dic[obs_name+'K_err'] = e_k
            dic[obs_name+'R'] = m_r
            dic[obs_name+'R_err'] = e_r
            dic[obs_name+'_lattice'] = {
                'L1': latt.L1,
                'L2': latt.L2,
                'a1': latt.a1,
                'a2': latt.a2
            }

    with open(os.path.join(directory, 'res.pkl'), 'wb') as f:
        pickle.dump(dic, f)
