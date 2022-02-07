#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 05:20:44 2020

@author: fassaad
"""

import numpy as np                         # Numerical library
from py_alf import ALF_source, Simulation  # Interface with ALF

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


sim = Simulation(ALF_source(branch='master'), 'Hubbard', sim_dict,
                 machine= 'gnu',
                 mpi    = False,
                 n_mpi  = 24)

sim.compile()
sim.run()
sim.analysis() 

# Load all analysis results in a single Pandas dataframe
res = sim.get_obs()

# Save all results in a single file
res.to_pickle('Hubbard_1D.pkl')
