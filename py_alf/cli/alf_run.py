#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helper script for compiling and running ALF."""
# pylint: disable=invalid-name

__author__ = "Fakher F. Assaad, and Jonas Schwab"
__copyright__ = "Copyright 2020-2022, The ALF Project"
__license__ = "GPL"

import os
import json
from argparse import ArgumentParser
from collections import OrderedDict

from py_alf import ALF_source, Simulation


def _get_arg_parser():
    parser = ArgumentParser(
        description='Helper script for compiling and running ALF.',
        )
    parser.add_argument(
        '--alfdir', default=os.getenv('ALF_DIR', './ALF'),
        help="Path to ALF directory. (default: os.getenv('ALF_DIR', './ALF')")
    parser.add_argument(
        '--sims_file', default='./Sims',
        help="File defining simulations parameters. Each line starts with "
             "the Hamiltonian name and a comma, after wich follows a dict "
             "in JSON format for the parameters. "
             "A line that says stop can be used to interrupt. "
             "(default: './Sims')")
    parser.add_argument(
        '--branch', default=None,
        help='Git branch to checkout.')
    parser.add_argument(
        '--machine', default='GNU',
        help="Machine configuration                (default: 'GNU')")
    parser.add_argument(
        '--mpi', action="store_true",
        help='mpi run')
    parser.add_argument(
        '--n_mpi', default=4,
        help='number of mpi processes              (default: 4)')
    parser.add_argument(
        '--mpiexec', default="mpiexec",
        help="Command used for starting a MPI run  (default: 'mpiexec')")
    parser.add_argument(
        '--mpiexec_args', default='',
        help='Additional arguments to MPI executable.')
    parser.add_argument(
        '--do_analysis', '--ana', action="store_true",
        help='Run default analysis after each simulation.')
    return parser


def _main():
    parser = _get_arg_parser()
    args = parser.parse_args()
    mpiexec_args = args.mpiexec_args.split()

    alf_dir = os.path.abspath(args.alfdir)

    alf_src = ALF_source(alf_dir=alf_dir, branch=args.branch)

    with open(args.sims_file, 'r', encoding="UTF-8") as f:
        simulations = f.read().splitlines()
    num_sims = 0
    ham_names = []
    sim_dicts = []
    for i, sim_str in enumerate(simulations):
        del i
        if sim_str.strip().lower() == "stop":
            break
        num_sims += 1
        ham_name, par_str = sim_str.split(sep=',', maxsplit=1)
        sim_dict = json.loads(par_str, object_pairs_hook=OrderedDict)
        ham_names.append(ham_name)
        sim_dicts.append(sim_dict)

    print("Number of simulations: {}".format(num_sims))
    for ham_name, sim_dict in zip(ham_names, sim_dicts):
        sim = Simulation(
            alf_src, ham_name, sim_dict,
            machine=args.machine, mpi=args.mpi, n_mpi=args.n_mpi,
            mpiexec=args.mpiexec, mpiexec_args=mpiexec_args
            )
        sim.compile()
        sim.run()
        if args.do_analysis:
            sim.analysis()
    print("Done")


if __name__ == '__main__':
    _main()
