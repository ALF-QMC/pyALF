#!/usr/bin/env python3
"""Created on Sat Aug 29 05:20:44 2020.

@author: fassaad
"""

from py_alf import ALF_source, Simulation  # Interface with ALF
from py_alf.ana import load_res  # Function for loading analysis results

sims = []                            # List of Simulation instances
time_steps = [0.0, 0.01]
for time_step in time_steps:
    if time_step == 0.0:
        sim_dict = {"Model": "Hubbard",
                    "Lattice_type": "N_leg_ladder",
                    "L1": 1 , "L2": 6,
                    "Beta": 4.0,
                    "Nsweep": 1000,
                    "NBin": 10,
                    "Ltau": 0,
                    "Dtau": 0.1,
                    "Delta_t_Langevin_HMC" :  time_step,
                    }
    else:
        sim_dict = {"Model": "Hubbard",
                    "Lattice_type": "N_leg_ladder",
                    "L1": 1 , "L2": 6,
                    "Beta": 4.0,
                    "Nsweep": 1000,
                    "NBin": 10,
                    "Ltau": 0,
                    "Dtau": 0.1,
                    "Continuous" : True,
                    "Langevin" : True,
                    "Delta_t_Langevin_HMC" :  time_step,
                    }
    sim = Simulation(
        #ALF_source(branch='165-introduce_langevin_updating_in_alf_2-0'),
        ALF_source(branch='master'),
        'Hubbard',
        sim_dict,
        machine= 'Intel',
        mpi    = True,
        n_mpi  = 12,
        )
    sim.sim_dir += "_TL={}".format(sim_dict["Delta_t_Langevin_HMC"])
    sims.append(sim)

# Compile ALF
sims[0].compile()

# Perform Monte Carlo simulations
for sim in sims:
    print (sim.sim_dir)
    sim.run()
    sim.analysis()

# Load all analysis results in a single Pandas dataframe
directories = [sim.sim_dir for sim in sims]
res = load_res(directories)

# Save all results in a single file
res.to_pickle('Langevin.pkl')

with open('Ener_Langevin.dat', 'w') as file:
    for i, time_step in zip(res.index, time_steps):
        item = res.loc[i]
        print(i)
        file.write(
            '{:6.6f}\t{:6.6f}\t{:6.6f}\n'.format(
                time_step,
                item['Ener_scal0'],
                item['Ener_scal0_err'],
            )
        )
