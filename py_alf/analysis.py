"""Supplies the default analysis routine."""
# pylint: disable=invalid-name
# pylint: disable=protected-access

import os
import pickle

import h5py
import numpy as np

from .ana import (
    AnalysisResultsScal,
    Parameters,
    ReadObs,
    ana_eq,
    ana_hist,
    ana_scal,
    ana_tau,
    custom_obs_get_dtype_len,
    error,
    write_res_eq,
    write_res_tau,
)
from .exceptions import AlreadyAnalyzed, TooFewBinsError


def _get_obs_list(directory, always=False):
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
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
                    raise AlreadyAnalyzed
            except OSError:
                pass

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
            raise TooFewBinsError
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
    return params, list_obs, list_scal, list_hist, list_eq, list_tau



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
        List of functions representing symmetry operations on lattice,
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
    print(f'### Analyzing {directory} ###')
    print(os.getcwd())

    try:
        dic, list_obs, list_scal, list_hist, list_eq, list_tau = \
            _get_obs_list(directory, always)
    except (AlreadyAnalyzed, TooFewBinsError):
        return

    if 'res' not in os.listdir(directory):
        os.mkdir(os.path.join(directory, 'res'))

    results_file = os.path.join(directory, 'results.h5')
    with h5py.File(results_file, "w") as f:
        f.attrs.create('program_name', 'ALF')

    if custom_obs is not None:
        print("Custom observables:")
        for obs_name, obs_spec in custom_obs.items():
            if all(x in list_obs for x in obs_spec['needs']):
                print('custom', obs_name, obs_spec['needs'])
                jacks = [ReadObs(directory, obs_name)
                         for obs_name in obs_spec['needs']]

                N_bins = jacks[0].N_bins
                dtype, length = custom_obs_get_dtype_len(obs_spec, jacks)
                shape = (N_bins,) if length == 1 else (N_bins, length)
                J = np.empty(shape, dtype=dtype)
                for i in range(N_bins):
                    J[i] = obs_spec['function'](
                        *[x for j in jacks for x in j.slice(i)],
                        **obs_spec['kwargs'])

                dat = error(J)
                results = AnalysisResultsScal(
                    "obs_name", np.nan, np.nan, dat[0], dat[1]
                    )
                results.write_to_hdf5(results_file)

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
            results = ana_scal(directory, obs_name)
        except TooFewBinsError:
            print("Too few bins, skipping.")
            continue

        results.write_to_hdf5(results_file)
        results.append_to_flat_dict(dic)
        results.savetxt(os.path.join(directory, 'res', obs_name))

    print("Histogram observables:")
    for obs_name in list_hist:
        print(obs_name)
        try:
            results = ana_hist(directory, obs_name)
        except TooFewBinsError:
            print("Too few bins, skipping.")
            continue
        dic[obs_name] = results

        np.savetxt(
            os.path.join(directory, 'res', obs_name),
            np.column_stack([results._values['dat'].value,
                             results._values['dat'].error]),
            header='Sign: {} {}, above {} {}, below {} {}'.format(
                results._values['sign'].value, results._values['sign'].error,
                results._values['above'].value, results._values['above'].error,
                results._values['below'].value, results._values['below'].error
            )
        )

    print("Equal time observables:")
    for obs_name in list_eq:
        print(obs_name)
        try:
            results = ana_eq(directory, obs_name, sym=symmetry)
        except TooFewBinsError:
            print("Too few bins, skipping.")
            continue
        results.write_to_hdf5(results_file)

        write_res_eq(
            directory, obs_name,
            results._values['k'].value, results._values['k'].error,
            results._values['k_traced'].value, results._values['k_traced'].error,
            results._values['r'].value, results._values['r'].error,
            results._values['r_traced'].value, results._values['r_traced'].error,
            results.latt
        )

        dic[obs_name+'K'] = results._values['k'].value
        dic[obs_name+'K_err'] = results._values['k'].error
        dic[obs_name+'K_sum'] = results._values['k_traced'].value
        dic[obs_name+'K_sum_err'] = results._values['k_traced'].error
        dic[obs_name+'R'] = results._values['r'].value
        dic[obs_name+'R_err'] = results._values['r'].error
        dic[obs_name+'R_sum'] = results._values['r_traced'].value
        dic[obs_name+'R_sum_err'] = results._values['r_traced'].error
        dic[obs_name+'_lattice'] = {
            'L1': results.latt.L1,
            'L2': results.latt.L2,
            'a1': results.latt.a1,
            'a2': results.latt.a2
        }

    if do_tau:
        print("Time displaced observables:")
        for obs_name in list_tau:
            print(obs_name)
            try:
                results = ana_tau(directory, obs_name, sym=symmetry)
            except TooFewBinsError:
                print("Too few bins, skipping.")
                continue
            results.write_to_hdf5(results_file)

            write_res_tau(
                directory, obs_name,
                results._values['k'].value, results._values['k'].error,
                results._values['r'].value, results._values['r'].error,
                results._values['dtau'].value,
                results.latt,
            )

            dic[obs_name+'K'] = results._values['k'].value
            dic[obs_name+'K_err'] = results._values['k'].error
            dic[obs_name+'R'] = results._values['r'].value
            dic[obs_name+'R_err'] = results._values['r'].error
            dic[obs_name+'_lattice'] = {
                'L1': results.latt.L1,
                'L2': results.latt.L2,
                'a1': results.latt.a1,
                'a2': results.latt.a2
            }

    with open(os.path.join(directory, 'res.pkl'), 'wb') as f:
        pickle.dump(dic, f)
