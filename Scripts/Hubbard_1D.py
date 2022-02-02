#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 05:20:44 2020

@author: fassaad
"""

import numpy as np                         # Numerical library
from py_alf import ALF_source, Simulation  # Interface with ALF

sims = []                                # List of Simulation instances
sim_dict = {"Model": "Hubbard", 
            "Lattice_type": "Square", 
            "L1": 14 , "L2": 1, 
            "Beta"      : 10.0,
            "Projector" : False, 
            "Theta"     : 10.0, 
            "Ham_U"     : 4.0,  
            "Ham_U2"    : 0.0,  
            "Ham_T"     : 1.0, 
            "Ham_T2"    : 0.0, 
            "ham_Tperp" : 0.0,  
            "Nsweep"    : 200, 
            "NBin"      : 5,
            "Ltau"      : 1,
            "Mz"        : False,
            }


sim = Simulation(ALF_source(), 'Hubbard', sim_dict,
                 branch = 'master',
                 machine= 'gnu',
                 mpi    = False,
                 n_mpi  = 24)
sims.append(sim)

sims[0].compile(target = "Hubbard")
 
for i, sim in enumerate(sims):
    print (sim.sim_dir)
    sim.run()
    print (sim.sim_dir)
    sim.analysis() 
