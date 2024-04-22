#!/usr/bin/env python3
"""
Created on Sat Aug 29 05:20:44 2020

@author: fassaad
"""

from py_alf import ALF_source, Simulation  # Interface with ALF

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
sim = Simulation(
    ALF_source(branch='master'),
    'Z2_Matter',
    sim_dict,
    machine='Intel',
    mpi=True,
    n_mpi=12,
    )

sim.compile()
sim.run()
sim.analysis()
res = sim.get_obs()

# Save all results in a single file
res.to_pickle('Z2_Matter.pkl')
