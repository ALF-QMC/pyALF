#!/usr/bin/env python
"""
Delete N bins in all observables of the specified HDF5-file.

Command line arguments:
   First argument: Name of HDF5 file
   Second argument: Number of first N0 bins to leave
   Second argument: Number of bins to remove after first N0 bins
"""

import sys
import h5py
import numpy as np


def reshape(fileobj, dset_name, N0, N):
    dset = fileobj[dset_name]
    dat = np.copy(np.concatenate([dset[:N0], dset[N0+N:]]))
    fileobj[dset_name].resize(dat.shape)
    fileobj[dset_name][:] = dat


def del_bins(filename, N0, N):
    with h5py.File(filename, 'r+') as f:
        for o in f:
            if o.endswith('_scal') or o.endswith('_eq') \
               or o.endswith('_tau') or o.endswith('_hist'):
                reshape(f, o+"/obser", N0, N)
                reshape(f, o+"/sign", N0, N)

            if o.endswith('_eq') or o.endswith('_tau'):
                reshape(f, o+"/back", N0, N)

            if o.endswith('_hist'):
                reshape(f, o+"/above", N0, N)
                reshape(f, o+"/below", N0, N)


if __name__ == '__main__':
    filename_in = sys.argv[1]
    N0_in = int(sys.argv[2])
    N_in = int(sys.argv[3])
    del_bins(filename_in, N0_in, N_in)
