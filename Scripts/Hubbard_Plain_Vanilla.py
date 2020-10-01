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
sim_dict = {"Model": "Hubbard_Plain_Vanilla", 
            "Lattice_type": "Square", 
            "L1": 4 , "L2": 1, 
            "Beta": 1.0, 
            "Nsweep": 2000, 
            "NBin": 40,
            "Ltau": 0,
            "Dtau": 0.05,
            "Projector" : True,
            }
sim = Simulation('Hubbard_Plain_Vanilla', sim_dict,
                 alf_dir= os.getenv('ALF_DIR', './ALF'),
                 branch = 'master',
                 machine= 'gnu',
                 )
sims.append(sim)

sims[0].compile(target = "Hubbard_Plain_Vanilla")
En = np.empty((len(sims), 2))
        
for i, sim in enumerate(sims):
    print (sim.sim_dir)
    sim.run()
    sim.analysis() 
    En[i] = sim.get_obs(['Ener_scalJ'])['Ener_scalJ']['obs']  # Store 


with open('Ener_Plain_Vanilla.dat', 'w') as file:   
    file.write('%6.6f\t' % (En[i,0])) 
    file.write('%6.6f\t' % (En[i,1]))
