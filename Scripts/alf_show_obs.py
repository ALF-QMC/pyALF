#!/usr/bin/env python3
"""Show observables and their number of bins in ALF HDF5 file(s).
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
    print("===== {} =====".format(filename))
    with h5py.File(filename,'r') as f:
        print("Scalar observables:")
        for o in f:
            if o.endswith('_scal'):
                N_bins = f[o+"/obser"].shape[0]
                print("{}; Bins: {}".format(o, N_bins))

        print("Histogram observables:")
        for o in f:
            if o.endswith('_hist'):
                N_bins = f[o+"/obser"].shape[0]
                print("{}; Bins: {}".format(o, N_bins))

        print("Equal time observables:")
        for o in f:
            if o.endswith('_eq'):
                N_bins = f[o+"/obser"].shape[0]
                print("{}; Bins: {}".format(o, N_bins))

        print("Time displaced observables:")
        for o in f:
            if o.endswith('_tau'):
                N_bins = f[o+"/obser"].shape[0]
                print("{}; Bins: {}".format(o, N_bins))
