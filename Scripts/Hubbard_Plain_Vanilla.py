#!/usr/bin/env python3
"""Created on Sat Aug 29 05:20:44 2020

@author: fassaad
"""

from py_alf import ALF_source, Simulation  # Interface with ALF

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
sim = Simulation(
    ALF_source(branch='master'),
    'Hubbard_Plain_Vanilla',
    sim_dict,
    machine= 'gnu',
    )

sim.compile()
sim.run()
sim.analysis()
res = sim.get_obs()

# Save all results in a single file
res.to_pickle('Hubbard_Plain_Vanilla.pkl')

with open('Ener_Plain_Vanilla.dat', 'w') as file:
    file.write('{:6.6f}\t{:6.6f}\n'.format(
        res.iloc[0]['Ener_scal0'],
        res.iloc[0]['Ener_scal0_err'],
        ))
