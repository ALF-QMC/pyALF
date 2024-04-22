#!/usr/bin/env python3
"""
Created on Sat Aug 29 05:20:44 2020

@author: fassaad
"""

import numpy as np  # Numerical library

from py_alf import ALF_source, Lattice, Simulation  # Interface with ALF
from py_alf.ana import load_res  # Function for loading analysis results

L = 8
alf_src = ALF_source(branch='master')
sims = []                                # List of Simulation instances
for Ham_V in [0.5, 0.6, 0.7, 0.8, 0.9,
              1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:  # Values of hybredization
    print(Ham_V)
    sim_dict = {"Model": "Hubbard",
                "Lattice_type": "Bilayer_square",
                "L1": L , "L2": L,
                "Beta"      : 1.0,
                "Projector" : True,
                "Theta"     : 10.0,
                "Ham_U"     : 0.0,
                "Ham_U2"    : 4.0,
                "Ham_T"     : 1.0,
                "Ham_T2"    : 0.0,
                "ham_Tperp" : Ham_V,
                "Nsweep"    : 200,
                "NBin"      : 20,
                "Ltau"      : 0,
                "Mz"        : False,
                }

    sim = Simulation(alf_src, 'Hubbard', sim_dict,
                     machine= 'Intel',
                     mpi    = True,
                     n_mpi  = 24,
                     )
    sims.append(sim)

sims[0].compile()
for sim in sims:
    sim.run()
    sim.analysis()

# Load all analysis results in a single Pandas dataframe
directories = [sim.sim_dir for sim in sims]
res = load_res(directories)

# Save all results in a single file
res.to_pickle(f'Hubbard_PAM_L{L}.pkl')

# Create lattice object
latt = Lattice(res.iloc[0]['SpinZ_eq_lattice'])
n = latt.k_to_n((np.pi, np.pi))  # Index of k=(pi,pi)

with open(f'Spin_PAM_L{L}.dat', 'w') as file:
    for i in res.index:
        item = res.loc[i]
        print(i)
        file.write(
            '{:6.6f}\t{:6.6f}\t{:6.6f}\n'.format(
                item['ham_tperp'],
                item['SpinZ_eqK'][1, 1, n],
                item['SpinZ_eqK_err'][1, 1, n]
                ))
