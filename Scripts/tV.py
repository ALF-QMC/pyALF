#!/usr/bin/env python3
"""
Created on Sat Aug 29 05:20:44 2020

@author: fassaad
"""

from py_alf import ALF_source, Simulation  # Interface with ALF

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
        

sim = Simulation(ALF_source(branch='master'), 'tV', sim_dict,
                 machine= 'gnu',
                 mpi    = False,
                 n_mpi  = 24)

sim.compile()
sim.run()
sim.analysis()
res = sim.get_obs()

# Save all results in a single file
res.to_pickle('tV.pkl')
