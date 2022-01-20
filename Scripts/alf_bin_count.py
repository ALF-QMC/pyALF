#!/usr/bin/env python3
"""Count number of bins in ALF HDF5 file(s), assuming all
observables have same number of bins.

Arguments: Name of HDF5 files.
If no arguments are supplied, all files named "data.h5" in the current working
directory and below are taken.
"""

__author__ = "Jonas Schwab"
__copyright__ = "Copyright 2022, The ALF Project"
__license__ = "GPL"

import sys
import os

from alf_ana.util import bin_count


if len(sys.argv) > 1:
    filenames = sys.argv[1:]
else:
    filenames = []
    for root, folders, files in os.walk('.'):
        if 'data.h5' in files:
            filenames.append(os.path.join(root, 'data.h5'))
    filenames.sort()

for filename in filenames:
    bin_count(filename)
