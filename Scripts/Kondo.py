#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 05:20:44 2020

@author: fassaad
"""

import numpy as np                                  # Numerical library
from py_alf import ALF_source, Simulation, Lattice  # Interface with ALF
from py_alf.ana import load_res            # Function for loading analysis results

alf_src = ALF_source(branch='master')
sims = []                                  # List of Simulation instances
for Ham_Uf in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
        1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]:  # Values of Uf
    print(Ham_Uf)
    sim_dict = {"Model": "Kondo",
                "Lattice_type": "Bilayer_square",
                "L1": 4 , "L2": 4,
                "Ham_Uf": Ham_Uf,
                "Beta": 5.0,
                #"Nsweep": 500,
                "Nsweep": 1,
                "NBin": 400,
                "Ltau": 0}
    sim = Simulation(alf_src, 'Kondo', sim_dict,
                     machine= 'Intel',
                     mpi    = True,
                     #n_mpi  = 12
                     n_mpi  = 4
                     )
    sims.append(sim)
sims[0].compile()        

for i, sim in enumerate(sims):
    sim.run()
    sim.analysis() 

# Load all analysis results in a single Pandas dataframe
directories = [sim.sim_dir for sim in sims]
res = load_res(directories)

# Save all results in a single file
res.to_pickle('Kondo.pkl')

# Create lattice object
latt = Lattice(res.iloc[0]['SpinZ_eq_lattice'])
n = latt.k_to_n((np.pi, np.pi))  # Index of k=(pi,pi)

with open('Constraint.dat', 'w') as file:
    for i in res.index:
        item = res.loc[i]
        print(i)
        file.write(
            '{:6.6f}\t{:6.6f}\t{:6.6f}\t{:6.6f}\t{:6.6f}\n'.format(
                item['ham_uf'],
                item['Constraint_scal0'],
                item['Constraint_scal0_err'],
                item['SpinZ_eqK'][1, 1, n],
                item['SpinZ_eqK_err'][1, 1, n]
                ))
