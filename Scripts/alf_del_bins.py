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

import sys

from alf_ana.util import del_bins


if __name__ == '__main__':
    filename_in = sys.argv[1]
    N0_in = int(sys.argv[2])
    N_in = int(sys.argv[3])
    del_bins(filename_in, N0_in, N_in)
