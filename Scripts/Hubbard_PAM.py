#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 05:20:44 2020

@author: fassaad
"""

import numpy as np                         # Numerical library
from py_alf import ALF_source, Simulation  # Interface with ALF

L=8
sims = []                                # List of Simulation instances
for Ham_V in [0.5,0.6,0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,1.5]:           # Values of hybredization    
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
        

    sim = Simulation(ALF_source(), 'Hubbard', sim_dict,           
                     branch = 'master',
                     machine= 'Intel',
                     mpi    = True,
                     n_mpi  = 24)
    sims.append(sim)

sims[0].compile(target = "Hubbard")
V   = np.empty((len(sims)))
Spin= np.empty((len(sims),L*L,2,2,4))
K   = np.empty((len(sims),L*L,2))           
for i, sim in enumerate(sims):
    print (sim.sim_dir)
    sim.run()
    print (sim.sim_dir)
    sim.analysis() 
    V[i] = sim.sim_dict['ham_Tperp']                             # Store V value
    Spin[i]= sim.get_obs(['SpinZ_eqJK'])['SpinZ_eqJK']['dat']
    K[i]   = sim.get_obs(['SpinZ_eqJK'])['SpinZ_eqJK']['k']


with open('Spin_PAM_L'+str(L)+'.dat', 'w') as file:
    for i in range(len(sims) ):
        file.write('%6.6f\t' % (V[i])) 
        file.write('%6.6f\t' % (Spin[i,L*L-1,1,1,0])) 
        file.write('%6.6f\t' % (Spin[i,L*L-1,1,1,1]))
        file.write('\n') 
