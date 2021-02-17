#!/usr/bin/env python3
"""Analyze Monte Carlo bins."""

import sys
import os
import pickle
import importlib.util

import numpy as np
import h5py

from alf_ana.ana import Parameters, ReadObs, error, ana_scal, ana_hist, ana_eq
from alf_ana.ana import ana_tau
# from alf_ana.custom_obs import c_obs, get_sym


def import_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ana(directory, sym_spec=None, custom_obs=None):
    print('### Analyzing {} ###'.format(directory))
    print(os.getcwd())

    par = Parameters(directory)
    if 'data.h5' in os.listdir(directory):
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

        with h5py.File(os.path.join(directory, 'data.h5'), "r") as f:
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

    dic = {}

    if custom_obs is not None:
        print("Custom observables:")
        for obs_name in custom_obs:
            func = custom_obs[obs_name][0]
            o_in = custom_obs[obs_name][1]
            kwarg = custom_obs[obs_name][2]
            if all(x in list_obs for x in o_in):
                print('custom', obs_name, o_in)
                jacks = [ReadObs(directory, obs_name) for obs_name in o_in]

                N_bins = jacks[0].N_bins
                dtype = func(*[x for j in jacks for x in j.slice(0)],
                             **kwarg).dtype
                J = np.empty(N_bins, dtype=dtype)
                for i in range(N_bins):
                    J[i] = custom_obs[obs_name][0](
                        *[x for j in jacks for x in j.slice(i)], **kwarg)

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
        sign, dat = ana_scal(directory, obs_name)

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
        sign, above, below, dat, upper, lower = ana_hist(directory, obs_name)

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

    # with open(os.path.join(directory,'res.pkl'), 'wb') as f:
        # pickle.dump(dic, f)
    # return

    print("Equal time observables:")
    for obs_name in list_eq:
        print(obs_name)
        if sym_spec is not None:
            symmetry = sym_spec(obs_name, par)
        else:
            symmetry = None
        sign, m_k, e_k, m_k_sum, e_k_sum, m_r, e_r, m_r_sum, e_r_sum, latt = \
            ana_eq(directory, obs_name, sym=symmetry)

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

    print("Time displaced observables:")
    for obs_name in list_tau:
        print(obs_name)
        if sym_spec is not None:
            symmetry = sym_spec(obs_name, par)
        else:
            symmetry = None
        sign, m_k, e_k, m_r0, e_r0, dtau, latt = \
            ana_tau(directory, obs_name, sym=symmetry)

        write_res_tau(directory, obs_name, m_k, e_k, m_r0, e_r0, dtau, latt)

        dic[obs_name+'K'] = m_k
        dic[obs_name+'K_err'] = e_k
        dic[obs_name+'R0'] = m_r0
        dic[obs_name+'R0_err'] = e_r0
        dic[obs_name+'_lattice'] = {
            'L1': latt.L1,
            'L2': latt.L2,
            'a1': latt.a1,
            'a2': latt.a2
            }

    with open(os.path.join(directory, 'res.pkl'), 'wb') as f:
        pickle.dump(dic, f)


def write_res_eq(directory, obs_name,
                 m_k, e_k, m_k_sum, e_k_sum,
                 m_r, e_r, m_r_sum, e_r_sum, latt):
    N_orb = m_k.shape[0]
    header = ['kx', 'ky']
    out = latt.k
    fmt = '\t'.join(['%8.5f %8.5f'] + ['% 13.10e % 13.10e']*(N_orb**2+1))
    fmth = '\t'.join(['{:^8s} {:^8s}'] + [' {:^33s}']*(N_orb**2+1))
    for no in range(N_orb):
        for no1 in range(N_orb):
            header = header + [str((no, no1))]
            out = np.column_stack([out, m_k[no1, no], e_k[no1, no]])
    out = np.column_stack([out, m_k_sum, e_k_sum])
    header = header + ['trace over n_orb']

    np.savetxt(
        os.path.join(directory, 'res', obs_name + '_K'),
        out,
        fmt=fmt,
        header=fmth.format(*header)
        )

    header = ['kx', 'ky']
    out = latt.k
    fmt = '\t'.join(['%8.5f %8.5f'] + ['% 13.10e % 13.10e']*(1))
    fmth = '\t'.join(['{:^8s} {:^8s}'] + [' {:^33s}']*(1))
    out = np.column_stack([out, m_k_sum, e_k_sum])
    header = header + ['trace over n_orb']
    np.savetxt(
        os.path.join(directory, 'res', obs_name + '_K_sum'),
        out,
        fmt=fmt,
        header=fmth.format(*header)
        )

    header = ['rx', 'ry']
    out = latt.r
    fmt = '\t'.join(['%9.5f %9.5f'] + ['% 13.10e % 13.10e']*(N_orb**2+1))
    fmth = '\t'.join(['{:^8s} {:^8s}'] + [' {:^33s}']*(N_orb**2+1))
    for no in range(N_orb):
        for no1 in range(N_orb):
            header = header + [str((no, no1))]
            out = np.column_stack([out, m_r[no1, no], e_r[no1, no]])
    out = np.column_stack([out, m_r_sum, e_r_sum])
    header = header + ['trace over n_orb']

    np.savetxt(
        os.path.join(directory, 'res', obs_name + '_R'),
        out,
        fmt=fmt,
        header=fmth.format(*header)
        )

    header = ['rx', 'ry']
    out = latt.r
    fmt = '\t'.join(['%9.5f %9.5f'] + ['% 13.10e % 13.10e']*(1))
    fmth = '\t'.join(['{:^8s} {:^8s}'] + [' {:^33s}']*(1))
    out = np.column_stack([out, m_r_sum, e_r_sum])
    header = header + ['trace over n_orb']
    np.savetxt(
        os.path.join(directory, 'res', obs_name + '_R_sum'),
        out,
        fmt=fmt,
        header=fmth.format(*header)
        )


def write_res_tau(directory, obs_name, m_k, e_k, m_r0, e_r0, dtau, latt):
    N_tau = m_k.shape[0]
    taus = np.linspace(0., (N_tau-1)*dtau, num=N_tau)

    for n in range(latt.N):
        directory2 = os.path.join(
            directory, 'res', obs_name, '{0:.2f}_{1:.2f}'.format(*latt.k[n]))
        if not os.path.exists(directory2):
            os.makedirs(directory2)

        np.savetxt(os.path.join(directory2, 'dat'),
                   np.column_stack([taus, m_k[:, n], e_k[:, n]]),
                   fmt=['%14.7f', '%16.8f', '%16.8f']
                   )

    np.savetxt(os.path.join(directory, 'res', obs_name, 'R0'),
               np.column_stack([taus, m_r0, e_r0]),
               fmt=['%14.7f', '%16.8f', '%16.8f']
               )


if __name__ == '__main__':
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    userspec_path = os.getenv('ALF_USERSPEC', None)
    if userspec_path is None:
        sym_spec = None
        custom_obs = None
    else:
        userspec = import_module('.', os.path.expanduser(userspec_path))
        sym_spec = userspec.get_sym
        custom_obs = userspec.c_obs

    if rank == 0:
        print(f'comm={comm}, size={size}, rank={rank}')
        if len(sys.argv) > 1:
            directories = sys.argv[1:]
        else:
            directories = []
            for root, folders, files in os.walk('.'):
                if 'data.h5' in files:
                    directories.append(root)
            directories.sort()

        data = [[] for i in range(size)]
        for i, directory in enumerate(directories):
            data[i % size].append(directory)
    else:
        data = None

    data = comm.scatter(data, root=0)

    for d in data:
        ana(d, sym_spec=sym_spec, custom_obs=custom_obs)
