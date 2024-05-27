#!/usr/bin/env python3
"""Example script showing the minimal steps for creating and running an ALF
simulation in pyALF.
"""
# pylint: disable=invalid-name

__author__ = "Jonas Schwab"
__copyright__ = "Copyright 2022, The ALF Project"
__license__ = "GPL"

# Import ALF_source and Simulation classes from the py_alf pythonmodule,
# which provide the interface with ALF."""
from py_alf import ALF_source, Simulation


def _main():
    # Create an instance of ALF_source, downloading the ALF source code from
    # https://git.physik.uni-wuerzburg.de/ALF/ALF, if ALF diretory does not exist.
    # Gets ALF diretory from environment variable $ALF_DIR, or defaults to "./ALF",
    # if not present.
    alf_src = ALF_source()

    # Create instance of `Simulation`, overwriting default parameters as desired.
    sim = Simulation(
        alf_src,
        "Hubbard",                    # Name of Hamiltonian
        {                             # Dictionary overwriting default parameters
            "Lattice_type": "Square"
        },
        machine='GNU'  # Change to "intel", or "PGI" if gfortran is not installed
    )

    # Compile ALF. The first time it will also download and compile HDF5,
    # which could take ~15 minutes.
    sim.compile()

    # Perform the simulation as specified in sim.
    sim.run()

    # Perform default analysis.
    sim.analysis()

    # Read analysis results into a Pandas Dataframe with one row per simulation,
    # containing parameters and observables.
    obs = sim.get_obs()
    print('Analysis results:')
    print(obs)

    print('Internal energy:')
    print(obs.iloc[0][['Ener_scal0', 'Ener_scal0_err',
                    'Ener_scal_sign', 'Ener_scal_sign_err']])

    # The simulation can be resumed by calling sim.run() again, increasing the
    # precision of results.
    sim.run()
    sim.analysis()
    obs = sim.get_obs()
    print('Internal energy:')
    print(obs.iloc[0][['Ener_scal0', 'Ener_scal0_err',
                    'Ener_scal_sign', 'Ener_scal_sign_err']])


if __name__ == '__main__':
    _main()
