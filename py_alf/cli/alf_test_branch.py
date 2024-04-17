#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helper script for testing between ALF branches."""
# pylint: disable=invalid-name

__author__ = "Jonas Schwab"
__copyright__ = "Copyright 2020-2022, The ALF Project"
__license__ = "GPL"

import os
import sys
import argparse
import json
import shutil
import tempfile

import numpy as np
from py_alf import ALF_source, Simulation


# def _test_branch(sim_name, alf_dir, pars, branch_R, branch_T, tmpdir,
#                 prepare_run=True, run=True, **kwargs):

def _create_sims(sim_name, alf_dir, pars, branch_R, branch_T, **kwargs):
    ham_name = pars[0]
    sim_dict = pars[1]

    sim_R = Simulation(
        ALF_source(alf_dir=alf_dir, branch=branch_R),
        ham_name, sim_dict, sim_dir=sim_name, **kwargs
        )
    sim_T = Simulation(
        ALF_source(alf_dir=alf_dir, branch=branch_T),
        ham_name, sim_dict, sim_dir=sim_name+'_test', **kwargs
        )
    return sim_R, sim_T

def _prepare_runs(tmpdir, sim_R, sim_T):
    executable_R = os.path.join(tmpdir, f'ALF_{sim_R.config.replace(" ", "_")}_reference.out')
    if os.path.isfile(executable_R):
        shutil.copy(executable_R, f'{sim_R.alf_src.alf_dir}/Prog/ALF.out')
    else:
        sim_R.compile()
        shutil.copy(f'{sim_R.alf_src.alf_dir}/Prog/ALF.out', executable_R)
    sim_R.run(copy_bin=True, only_prep=True)

    executable_T = os.path.join(tmpdir, f'ALF_{sim_T.config.replace(" ", "_")}_test.out')
    if os.path.isfile(executable_T):
        shutil.copy(executable_T, f'{sim_T.alf_src.alf_dir}/Prog/ALF.out')
    else:
        sim_T.compile()
        shutil.copy(f'{sim_T.alf_src.alf_dir}/Prog/ALF.out', executable_T)
    sim_T.run(copy_bin=True, only_prep=True)


def _analyze(sim_R, sim_T):
    sim_R.analysis()
    obs_R = sim_R.get_obs()
    sim_T.analysis()
    obs_T = sim_T.get_obs()
    # test_all = obs_R.equals(obs_T)
    with open(f'{sim_R.sim_dir}.txt', 'w', encoding='UTF-8') as f:
        test_all = True
        for name in obs_R:
            test = True
            for dat_R, dat_T in zip(obs_R[name], obs_T[name]):
                try:
                    test_temp = np.allclose(dat_R, dat_T)
                except TypeError:
                    pass
                test = test and test_temp
            f.write('{name}: {test}\n')
            test_all = test_all and test

    return test_all


def _get_arg_parser():
    parser = argparse.ArgumentParser(
        description='Script for testing two branches against one another.'
            'The test succeeds if analysis results for both branches are '
            'exactly the same.',
        )
    parser.add_argument(
        '--sim_pars', default='./test_pars.json',
        help="JSON file containing parameters for testing. "
             "(default: './test_pars.json')")
    parser.add_argument(
        '--alfdir', default=os.getenv('ALF_DIR', './ALF'),
        help="Path to ALF directory. (default: os.getenv('ALF_DIR', './ALF'))")
    parser.add_argument(
        '--branch_R', default="master",
        help='Reference branch.                      (default: master)')
    parser.add_argument(
        '--branch_T', default="master",
        help='Branch to test.                        (default: master)')
    parser.add_argument(
        '--machine', default="GNU",
        help='Machine configuration.                 (default: "GNU")')
    parser.add_argument(
        '--devel', action='store_true',
        help='Compile with additional flags for development and debugging.')
    parser.add_argument(
        '--mpi', action='store_true',
        help='Do MPI run(s).                         (default: False)')
    parser.add_argument(
        '--n_mpi', default=4,
        help='Number of MPI processes.               (default: 4)')
    parser.add_argument(
        '--mpiexec', default="mpiexec",
        help='Command used for starting an MPI run.  (default: "mpiexec")')
    parser.add_argument(
        '--mpiexec_args', default='',
        help='Additional arguments to MPI executable.')
    parser.add_argument(
        '--no_prep', action='store_true',
        help='Do not prepare runs, i.e. Compiling and creating directories.')
    parser.add_argument(
        '--no_sim', action='store_true',
        help='Do not run ALF binary.')
    parser.add_argument(
        '--no_analyze', action='store_true',
        help='Do not analyze and compare results.')
    return parser


def _main():
    parser = _get_arg_parser()
    args = parser.parse_args()

    alf_dir = os.path.abspath(args.alfdir)
    mpiexec_args = args.mpiexec_args.split()

    with open(args.sim_pars, 'r', encoding='UTF-8') as f:
        sim_pars = json.load(f)

    if os.path.exists("test.txt"):
        os.remove("test.txt")

    test_all = True
    tmpdir = tempfile.mkdtemp()
    print(f'Caching executables in {tmpdir}.')
    for sim_name, sim_dict in sim_pars.items():
        sim_R, sim_T = _create_sims(
            sim_name, alf_dir, sim_dict, args.branch_R, args.branch_T,
            machine=args.machine, mpi=args.mpi, n_mpi=args.n_mpi,
            mpiexec=args.mpiexec, mpiexec_args=mpiexec_args,
            devel=args.devel)
        if not args.no_prep:
            _prepare_runs(tmpdir, sim_R, sim_T)
        if not args.no_sim:
            sim_R.run(bin_in_sim_dir=True)
            sim_T.run(bin_in_sim_dir=True)
        if not args.no_analyze:
            test = _analyze(sim_R, sim_T)
            with open('test.txt', 'a', encoding='UTF-8') as f:
                f.write(f'{sim_name}: {test}\n')
            if not test:
                test_all = False
    os.removedirs(tmpdir)
    if not args.no_analyze:
        with open('test.txt', 'a', encoding='UTF-8') as f:
            f.write(f'\tTotal: {test_all}\n')
        if test_all:
            print("Test sucessful")
            sys.exit(0)
        else:
            print("Test failed")
            with open('test.txt', 'r', encoding='UTF-8') as f:
                print(f.read())
            sys.exit(1)


if __name__ == '__main__':
    _main()
