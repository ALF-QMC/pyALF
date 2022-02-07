#!/usr/bin/env python3
"""
Delete N bins in all observables of the specified HDF5-file.

Command line arguments:
   First argument: Name of HDF5 file
   Second argument: Number of first N0 bins to leave
   Third argument: Number of bins to remove after first N0 bins
"""

__author__ = "Jonas Schwab"
__copyright__ = "Copyright 2022, The ALF Project"
__license__ = "GPL"

from argparse import ArgumentParser

from py_alf.utils import del_bins


def _get_arg_parser():
    parser = ArgumentParser(
        description='Delete N bins in all observables of '
                    'the specified HDF5-file.',
        )
    parser.add_argument(
        '--N', type=int, required=True,
        help='Number of bins to remove after first N0 bins.')
    parser.add_argument(
        '--N0', type=int, default=0,
        help='Number of first N0 bins to leave. (default=0)')
    parser.add_argument(
        'filename', nargs=1,
        help='Name of HDF5 file..')
    return parser


if __name__ == '__main__':
    parser = _get_arg_parser()
    args = parser.parse_args()
    del_bins(args.filename, args.N0, args.N)
