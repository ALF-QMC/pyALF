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
sim_dict = {"Model": "LRC", 
            "Lattice_type": "Square", 
            "L1": 4 , "L2": 4, 
            "Beta": 5.0, 
            "Nsweep": 200, 
            "NBin": 50,
            "Ltau": 0,
            "Global_tau_moves" : True,
            "ham_U"            : 4.0 ,  
            "ham_alpha"        : 0.0 ,
            "Percent_change"   : 0.1      
            }
sim = Simulation('LRC', sim_dict,
                 alf_dir= os.getenv('ALF_DIR', './ALF'),
                 branch = 'master',
                 machine= 'gnu',
                 )
sims.append(sim)
sims[0].compile(target = "LRC")

En = np.empty((len(sims), 2))
        
for i, sim in enumerate(sims):
    print (sim.sim_dir)
    sim.run()
    sim.analysis() 
    En[i] = sim.get_obs(['Ener_scalJ'])['Ener_scalJ']['obs']  # Store 


with open('Ener_LRC.dat', 'w') as file:   
    file.write('%6.6f\t' % (En[i,0])) 
    file.write('%6.6f\t' % (En[i,1]))

#Con = np.empty((len(sims), 2))         # Matrix for storing energy values
#Uf  = np.empty((len(sims),))
#Spin= np.empty((len(sims),16,2,2,4))
#K   = np.empty((len(sims),16,2))           
#for i, sim in enumerate(sims):
#    print (sim.sim_dir)
#    sim.run()
#    print (sim.sim_dir)
#    sim.analysis() 
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
