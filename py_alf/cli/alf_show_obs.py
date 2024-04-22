#!/usr/bin/env python3
"""Show observables and their number of bins in ALF HDF5 file(s).

Arguments: Name of HDF5 files.
If no arguments are supplied, all files named "data.h5" in the current working
directory and below are taken.
"""

__author__ = "Jonas Schwab"
__copyright__ = "Copyright 2022, The ALF Project"
__license__ = "GPL"

import os
from argparse import ArgumentParser

from py_alf.utils import find_sim_dirs, show_obs


def _get_arg_parser():
    parser = ArgumentParser(
        description='Show observables and their number of bins '
                    'in ALF HDF5 file(s).',
        )
    parser.add_argument(
        'filenames', nargs='*',
        help='Name of HDF5 files. If no arguments are supplied, '
             'all files named "data.h5" in the current working '
             'directory and below are taken.')
    return parser


def _main():
    parser = _get_arg_parser()
    args = parser.parse_args()
    if args.filenames:
        filenames = args.filenames
    else:
        dirs = find_sim_dirs('.')
        filenames = [os.path.join(d, 'data.h5') for d in dirs]

    for filename in filenames:
        print(f"===== {filename} =====")
        show_obs(filename)


if __name__ == '__main__':
    _main()
