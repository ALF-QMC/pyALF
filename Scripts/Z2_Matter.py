#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 05:20:44 2020

@author: fassaad
"""

import os
from py_alf import Simulation            # Interface with ALF
import numpy as np                       # Numerical library
sims = []                                # Vector of Simulation instances
sim_dict = {"Model": "Z2_Matter", 
            "Lattice_type": "Square", 
            "L1": 8 , "L2": 8, 
            "Beta": 20.0, 
            "Dtau": 0.05, 
            "Nsweep": 200, 
            "NBin": 100,
            "Ltau": 0,
            "Ham_T": 0.0,
            "Ham_h": 0.0,
            "Ham_TZ2" : 1.0, 
            "Ham_K"   : 0.0, 
            "Ham_J"   : 0.0,
            "Ham_g"   : 1.0,
            "Ham_U"   : 0.0,
            "Global_tau_moves"   : True,
            "Propose_S0"         : False, 
            "Nwrap"   : 10,
            }
sim = Simulation('Z2_Matter', sim_dict,
                 alf_dir= os.getenv('ALF_DIR', './ALF'),
                 branch = 'master', 
                 machine= 'Intel',
                 mpi    = True,
                 n_mpi  = 12 ,
                 )
sims.append(sim)

sims[0].compile(target = "Z2_Matter")
#Con = np.empty((len(sims), 2))         # Matrix for storing energy values
#Uf  = np.empty((len(sims),))
#Spin= np.empty((len(sims),16,2,2,4))
#K   = np.empty((len(sims),16,2))           
for i, sim in enumerate(sims):
     print (sim.sim_dir)
     sim.run()
     print (sim.sim_dir)
     sim.analysis() 
#    Uf[i] = sim.sim_dict['Ham_Uf']                             # Store Uf value
#    Con[i] = sim.get_obs(['Constraint_scalJ'])['Constraint_scalJ']['obs']  # Store constraint
#    Spin[i]= sim.get_obs(['SpinZ_eqJK'])['SpinZ_eqJK']['dat']
#    K[i]   = sim.get_obs(['SpinZ_eqJK'])['SpinZ_eqJK']['k']
#
#
#with open('Constraint.dat', 'w') as file:
#    for i in range(len(sims) ):
#        file.write('%6.6f\t' % (Uf[i])) 
#        file.write('%6.6f\t' % (Con[i,0])) 
#        file.write('%6.6f\t' % (Con[i,1])) 
#        file.write('%6.6f\t' % (Spin[i,15,1,1,0])) 
#        file.write('%6.6f\t' % (Spin[i,15,1,1,1]))
#        file.write('\n') 
