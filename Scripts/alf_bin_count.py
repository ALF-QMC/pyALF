#!/usr/bin/env python3
"""Count number of bins in ALF HDF5 file.


"""
import sys
import os
import h5py

if len(sys.argv) > 1:
    filenames = sys.argv[1:]
else:
    filenames = []
    for root, folders, files in os.walk('.'):
        if 'data.h5' in files:
            filenames.append(os.path.join(root, 'data.h5'))
    filenames.sort()

for filename in filenames:
    f = h5py.File(filename,'r')
    N_bins = 0
    for o in f:
        if '_scal' in o:
            N_bins = f[o +"/obser"].shape[0]
            break
    print(filename, N_bins)
