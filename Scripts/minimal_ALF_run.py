#!/usr/bin/env python3
"""
Example script showing the minimal steps for creating and running an ALF
simulation in pyALF.
"""

"""Import ALF_source and Simulation classes from the py_alf pythonmodule,
which provide the interface with ALF."""
from py_alf import ALF_source, Simulation


"""Create an instance of ALF_source, downloading the ALF source code from
https://git.physik.uni-wuerzburg.de/ALF/ALF, if alf_dir does not exist."""
alf_src = ALF_source(
    alf_dir='./ALF',
    branch='196-write-parameters-to-hdf5-file',  # TODO: Remove this after merging '196-write-parameters-to-hdf5-file' into master
)

"""Create an instance of `Simulation`, overwriting default parameters as
desired."""
sim = Simulation(
    alf_src,
    "Hubbard",                    # Name of Hamiltonian
    {                             # Dictionary overwriting default parameters
        "Lattice_type": "Square"
    },
)

"""Compile ALF. The first time it will also download and compile HDF5,
which could take ~15 minutes."""
sim.compile()

"""Perform the simulation as specified in sim."""
sim.run()

"""Perform some simple analysis."""
sim.analysis()

"""Read analysis results into a Pandas Dataframe with one row per simulation,
containing parameters and observables:"""
obs = sim.get_obs()
print('Analysis results:')
print(obs)

print('Internal energy:')
print(obs.iloc[0][['Ener_scal0', 'Ener_scal0_err',
                   'Ener_scal_sign', 'Ener_scal_sign_err']])

"""The simulation can be resumed by calling sim.run() again, increasing the
precision of results."""
sim.run()
sim.analysis()
obs = sim.get_obs()
print('Internal energy:')
print(obs.iloc[0][['Ener_scal0', 'Ener_scal0_err',
                   'Ener_scal_sign', 'Ener_scal_sign_err']])
