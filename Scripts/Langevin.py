#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 05:20:44 2020

@author: fassaad
"""

import os
from py_alf import Simulation            # Interface with ALF
import numpy as np                       # Numerical library
sims = []      
for Time_step in [0.01, 0.02]:                          # Vector of Simulation instances
    sim_dict = {"Model": "Hubbard", 
                "Lattice_type": "N_leg_ladder", 
                "L1": 1 , "L2": 6, 
                "Beta": 1.0, 
                "Nsweep": 100, 
                "NBin": 10,
                "Ltau": 0,
                "Dtau": 0.05,
                "Projector" : True,
                "Continuous" : True, 
                "Global_update_scheme" : "Langevin",
                "Delta_t_Langevin_HMC" :  Time_step, 
                }
    sim = Simulation('Hubbard', sim_dict,
                     #alf_dir= os.getenv('ALF_DIR', './ALF'),
                     alf_dir='/Users/fassaad/Programs/ALF/Work',
                     branch = '165-introduce_langevin_updating_in_alf_2-0',
                     machine= 'gnu',
                     )
    #sim.sim_dir += 'Time_step=' + str(dict["Time_step"])
    sim.sim_dir += "_TL={}".format(sim_dict["Delta_t_Langevin_HMC"])
    sims.append(sim)

#sims[0].compile(target = "Hubbard")

En = np.empty((len(sims), 2))
TS = np.empty((len(sims), 1))
        
for i, sim in enumerate(sims):
    print (sim.sim_dir)
    #sim.run()
    sim.analysis() 
    En[i] = sim.get_obs(['Ener_scalJ'])['Ener_scalJ']['obs'] 
    TS[i] = sim.sim_dict["Delta_t_Langevin_HMC"]  

with open('Ener_Langevin.dat', 'w') as file:
    for i in range(len(sims) ):
        file.write('%6.6f\t' % (TS[i])) 
        file.write('%6.6f\t' % (En[i,0]))   
        file.write('%6.6f\t' % (En[i,1]))
        file.write('\n')
