#!/usr/bin/env python3
"""Analyze Monte Carlo bins."""
# pylint: disable=invalid-name

import sys
import os
import importlib.util

from alf_ana.ana import ana


def import_module(module_name, path):
    """Dynamically import module from given path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()

    userspec_path = os.getenv('ALF_USERSPEC', None)
    if userspec_path is None:
        sym_spec = None
        custom_obs = None
    else:
        userspec = import_module('.', os.path.expanduser(userspec_path))
        sym_spec = userspec.get_sym
        custom_obs = userspec.c_obs

    if RANK == 0:
        print('comm={}, size={}, rank={}'.format(COMM, SIZE, RANK))
        if len(sys.argv) > 1:
            directories = sys.argv[1:]
        else:
            directories = []
            for root, folders, files in os.walk('.'):
                if 'data.h5' in files:
                    directories.append(root)
            directories.sort()

        data = [[] for i in range(SIZE)]
        for i, directory in enumerate(directories):
            data[i % SIZE].append(directory)
    else:
        data = None

    data = COMM.scatter(data, root=0)

    for d in data:
        ana(d, sym_spec=sym_spec, custom_obs=custom_obs)
