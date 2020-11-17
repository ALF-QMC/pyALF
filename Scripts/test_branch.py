#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper script for testing between ALF branches.
"""
# pylint: disable=invalid-name

__author__ = "Jonas Schwab"
__copyright__ = "Copyright 2020, The ALF Project"
__license__ = "GPL"

import os
import sys
import argparse
import numpy as np
from py_alf import Simulation


def test_branch(alf_dir, sim_dict, branch_R, branch_T,
                machine="DEVELOPMENT", mpi=False, n_mpi=4):
    ham_name = sim_dict.pop("ham_name", sim_dict["Model"])
    sim_R = Simulation(ham_name, sim_dict, alf_dir,
                       machine=machine,
                       branch=branch_R,
                       mpi=mpi,
                       n_mpi=n_mpi)
    sim_R.compile(target=ham_name)
    sim_R.run()
    sim_R.analysis()
    obs_R = sim_R.get_obs()

    sim_T = Simulation(ham_name, sim_dict, alf_dir,
                       machine=machine,
                       branch=branch_T,
                       mpi=mpi,
                       n_mpi=n_mpi)
    sim_T.sim_dir = sim_T.sim_dir + '_test'
    sim_T.compile(target=ham_name)
    sim_T.run()
    sim_T.analysis()
    obs_T = sim_T.get_obs()

    test_all = True
    with open(f'{sim_R.sim_dir}.txt', 'w') as f:
        for name in obs_R:
            if name.endswith('_scalJ'):
                x_R = obs_R[name]['obs']
                x_T = obs_T[name]['obs']
                test = np.allclose(x_R, x_T)
                f.write(f'{name}: {test}\n')
                f.write(f'    reference: {x_R}\n')
                f.write(f'         test: {x_T}\n')
                if not test:
                    test_all = False
        for name in obs_R:
            if name.endswith('_eqJK') or name.endswith('_eqJR'):
                test = np.allclose(obs_R[name]['dat'], obs_T[name]['dat'])
                f.write(f'{name}: {test}\n')
                if not test:
                    test_all = False
    return test_all


sim_pars = {
    "Hubbard_1D": {
        "ham_name": "Hubbard",
        "Model": "Hubbard",
        "Lattice_type": "Square",
        "L1": 6, "L2": 2,
        "Beta": 10.0,
        "Projector": False,
        "Theta": 10.0,
        "Ham_U": 4.0,
        "Ham_U2": 0.0,
        "Ham_T": 1.0,
        "Ham_T2": 0.0,
        "ham_Tperp": 0.0,
        "Nsweep": 20,
        "NBin": 5,
        "Ltau": 1,
        "Mz": False,
        },
    "Hubbard_PAM": {
        "ham_name": "Hubbard",
        "Model": "Hubbard",
        "Lattice_type": "Bilayer_square",
        "L1": 4, "L2": 4,
        "Beta": 1.0,
        "Projector": True,
        "Theta": 10.0,
        "Ham_U": 0.0,
        "Ham_U2": 4.0,
        "Ham_T": 1.0,
        "Ham_T2": 0.0,
        "ham_Tperp": 1.0,
        "Nsweep": 20,
        "NBin": 5,
        "Ltau": 1,
        "Mz": False,
        },
    "Hubbard_Plain_Vanilla": {
        "ham_name": "Hubbard_Plain_Vanilla",
        "Model": "Hubbard_Plain_Vanilla",
        "Lattice_type": "Square",
        "L1": 4, "L2": 4,
        "Beta": 1.0,
        "Nsweep": 20,
        "NBin": 5,
        "Ltau": 1,
        "Dtau": 0.05,
        "Projector": True,
        },
    "Kondo": {
        "ham_name": "Kondo",
        "Model": "Kondo",
        "Lattice_type": "Bilayer_square",
        "L1": 4, "L2": 4,
        "Ham_Uf": 1.0,
        "Beta": 5.0,
        "Nsweep": 20,
        "NBin": 5,
        "Ltau": 1
        },
    "LRC": {
        "ham_name": "LRC",
        "Model": "LRC",
        "Lattice_type": "Square",
        "L1": 4, "L2": 4,
        "Beta": 5.0,
        "Nsweep": 20,
        "NBin": 5,
        "Ltau": 0,
        "Global_tau_moves": True,
        "ham_U": 4.0,
        "ham_alpha": 0.0,
        "Percent_change": 0.1
        },
    "tV": {
        "ham_name": "tV",
        "Model": "tV",
        "Lattice_type": "Square",
        "L1": 4, "L2": 4,
        "Beta": 10.0,
        "Projector": False,
        "Theta": 10.0,
        "Ham_T": 1.0,
        "Ham_T2": 0.0,
        "Ham_Tperp": 0.0,
        "Ham_V": -1.0,
        "Ham_V2": 0.0,
        "Ham_Vperp": 0.0,
        "Nsweep": 20,
        "NBin": 5,
        "Ltau": 0,
        "N_SUN": 1,
        "Global_tau_moves": False,
        },
    "Z2_Matter": {
        "ham_name": "Z2_Matter",
        "Model": "Z2_Matter",
        "Lattice_type": "Square",
        "L1": 4, "L2": 4,
        "Beta": 20.0,
        "Dtau": 0.05,
        "Nsweep": 20,
        "NBin": 10,
        "Ltau": 0,
        "Ham_T": 0.0,
        "Ham_h": 0.0,
        "Ham_TZ2": 1.0,
        "Ham_K": 0.0,
        "Ham_J": 0.0,
        "Ham_g": 1.0,
        "Ham_U": 0.0,
        "Global_tau_moves": True,
        "Propose_S0": False,
        "Nwrap": 10,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script for testing two branches against one another',
        )
    parser.add_argument(
        '--alfdir', default=os.getenv('ALF_DIR', './ALF'),
        help="Path to ALF directory")
    parser.add_argument(
        '--branch_R', default="master",
        help='Reference branch.      (default: master)')
    parser.add_argument(
        '--branch_T', default="master",
        help='Branch to test.    (default: master)')
    parser.add_argument(
        '--machine', default="DEVELOPMENT",
        help='Machine configuration                (default: DEVELOPMENT)')
    parser.add_argument(
        '--mpi', default=False,
        help='mpi run                              (default: False)')
    parser.add_argument(
        '--n_mpi', default=4,
        help='number of mpi processes              (default: 4)')

    args = parser.parse_args()

    alf_dir = os.path.abspath(args.alfdir)
    branch_R = args.branch_R
    branch_T = args.branch_T
    machine = args.machine
    mpi = args.mpi
    n_mpi = args.n_mpi

    if os.path.exists("test.txt"):
        os.remove("test.txt")

    test_all = True
    for sim_name, sim_dict in sim_pars.items():
        test = test_branch(alf_dir, sim_dict, branch_R, branch_T, machine, mpi, n_mpi)
        with open('test.txt', 'w+') as f:
            f.write(f'{sim_name}: {test}\n')
        if not test:
            test_all = False
    with open('test.txt', 'w+') as f:
        f.write(f'\tTotal: {test_all}\n')
    if test_all:
        print("Test sucessful")
        sys.exit(0)
    else:
        print("Test failed")
        with open('test.txt', 'r') as f:
            print(f.read())
        sys.exit(0)
