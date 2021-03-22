#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper script for compiling, running and testing ALF.
"""
# pylint: disable=invalid-name

__author__ = "Fakher F. Assaad, and Jonas Schwab"
__copyright__ = "Copyright 2020, The ALF Project"
__license__ = "GPL"

import os
import sys
import json
import argparse
from collections import OrderedDict
from py_alf import Simulation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Helper script for compiling, running and testing ALF.',
        )
    parser.add_argument(
        '--alfdir', required=True,
        help="Path to ALF directory")
    parser.add_argument(
        '--ham_name_R', required=True,
        help='Name of hamiltonian')
    parser.add_argument(
        '-R', action='store_true',
        help='Do a run')
    parser.add_argument(
        '-T', action='store_true',
        help='Do a test')
    parser.add_argument(
        '--branch_R', default="master",
        help='Git branch to checkout for run.      (default: master)')
    parser.add_argument(
        '--branch_T', default="master",
        help='Branch to test against Runbranch.    (default: master)')
    parser.add_argument(
        '--machine', default='GNU',
        help='Machine configuration                (default: GNU)')
    parser.add_argument(
        '--ham_name_T',
        help='Name of test hamiltonian             (default: ham_name_R)')
    parser.add_argument(
        '--mpi', default=False,
        help='mpi run                              (default: False)')
    parser.add_argument(
        '--n_mpi', default=4,
        help='number of mpi processes              (default: 4)')

    args = parser.parse_args()

    do_R = args.R
    do_T = args.T
    alf_dir = os.path.abspath(args.alfdir)
    branch_R = args.branch_R
    branch_T = args.branch_T
    machine = args.machine
    ham_name_R = args.ham_name_R
    if args.ham_name_T is not None:
        ham_name_T = args.ham_name_T
    else:
        ham_name_T = args.ham_name_R
    mpi = args.mpi
    n_mpi = args.n_mpi

    with open("Sims") as f:
        simulations = f.read().splitlines()
    print("Number of simulations ", len(simulations))
    for sim in simulations:
        if sim.strip() == "stop":
            print("Done")
            sys.exit()
        sim_dict = json.loads(sim, object_pairs_hook=OrderedDict)
        if do_R:
            print('do R')
            sim_R = Simulation(ham_name_R, sim_dict, alf_dir,
                               machine=machine,
                               branch=branch_R, mpi=mpi, n_mpi=n_mpi)
            sim_R.compile()
            sim_R.run()
            sim_R.analysis()
            obs_R = sim_R.get_obs()
        if do_T:
            print('do T')
            sim_T = Simulation(ham_name_T, sim_dict, alf_dir,
                               machine=machine,
                               branch=branch_T, mpi=mpi, n_mpi=n_mpi)
            sim_T.sim_dir = sim_T.sim_dir + '_test'
            sim_T.compile()
            sim_T.run()
            sim_T.analysis()
            obs_T = sim_T.get_obs()

        # with open(sim_R.sim_dir + ".txt", "w") as f:
        #     f.write('Run:  {} +- {}\n'.format(*obs_R['Kin_scalJ']['obs'][0]))
        #     f.write('Test: {} +- {}\n'.format(*obs_T['Kin_scalJ']['obs'][0]))
        #     f.write('Diff: {} +- {}\n'.format(
        #         *(obs_R['Kin_scalJ']['obs'][0] - obs_T['Kin_scalJ']['obs'][0])
        #         ))
