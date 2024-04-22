#!/usr/bin/env python3
"""
Created on Sat Aug 29 05:20:44 2020

@author: fassaad
"""

from py_alf import ALF_source, Simulation  # Interface with ALF

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
sim = Simulation(
    ALF_source(branch='master'),
    'LRC',
    sim_dict,
    machine= 'gnu',
    )
sim.compile()
sim.run()
sim.analysis()
res = sim.get_obs()

# Save all results in a single file
res.to_pickle('LRC.pkl')

with open('Ener_LRC.dat', 'w') as file:
    file.write('{:6.6f}\t{:6.6f}\n'.format(
        res.iloc[0]['Ener_scal0'],
        res.iloc[0]['Ener_scal0_err'],
        ))
