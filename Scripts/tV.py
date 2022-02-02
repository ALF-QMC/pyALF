#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 05:20:44 2020

@author: fassaad
"""

import numpy as np                         # Numerical library
from py_alf import ALF_source, Simulation  # Interface with ALF

sims = []                                # Vector of Simulation instances
sim_dict = {"Model": "tV", 
            "Lattice_type": "Square", 
            "L1": 14 , "L2": 1, 
            "Beta"      : 10.0,
            "Projector" : False, 
            "Theta"     : 10.0,
            "Ham_T"     : 1.0, 
            "Ham_T2"    : 0.0,
            "Ham_Tperp" : 0.0,
            "Ham_V"     : -2.0,  
            "Ham_V2"    : 0.0, 
            "Ham_Vperp" : 0.0, 
            "Nsweep"    : 20, 
            "NBin"      : 5,
            "Ltau"      : 0,
            "N_SUN"     : 1, 
            }
        

sim = Simulation(ALF_source(), 'tV', sim_dict,
                 branch = 'master',
                 machine= 'gnu',
                 mpi    = False,
                 n_mpi  = 24)
sims.append(sim)

sims[0].compile(target = "tV")
 
for i, sim in enumerate(sims):
    print (sim.sim_dir)
    sim.run()
    print (sim.sim_dir)
    sim.analysis() 
