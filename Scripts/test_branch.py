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
import numpy as np
from py_alf import Simulation


def test_branch(alf_dir, ham_name, sim_dict, branch_R, branch_T,
                machine="DEVELOPMENT", mpi=False, n_mpi=4):
    sim_R = Simulation(ham_name, sim_dict, alf_dir,
                       machine=machine,
                       branch=branch_R,
                       mpi=mpi,
                       n_mpi=n_mpi)
    sim_R.compile(target=ham_name)
    sim_R.run()
    sim_R.analysis(legacy=True)
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

    with open(f'{sim_R.sim_dir}.txt', 'w') as f:
        for name in obs_R:
            if name.endswith('_scalJ'):
                test = np.allclose(obs_R[name]['obs'], obs_T[name]['obs'],
                                   )
            if name.endswith('_eqJK') or name.endswith('_eqJR'):
                test = np.allclose(obs_R[name]['dat'], obs_T[name]['dat'],
                                   )
            f.write(f'{name}: {test}\n')


if __name__ == "__main__":
    alf_dir = os.getenv('ALF_DIR', './ALF')
    branch_R = "master"
    branch_T = "122-embedding-lattice-information-in-observables"
    machine = "DEVELOPMENT"
    ham_names = ["Z2_Matter", "Hubbard_Plain_Vanilla", "Hubbard", "LRC",
                 "Kondo", "tV"]

    for ham_name in ham_names:
        test_branch(alf_dir, ham_name, {'Model': ham_name}, branch_R, branch_T)
