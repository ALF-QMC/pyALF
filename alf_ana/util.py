"""Utility functions for handling ALF HDF5 files."""
# pylint: disable=invalid-name
# pylint: disable=consider-using-f-string

__author__ = "Jonas Schwab"
__copyright__ = "Copyright 2022, The ALF Project"
__license__ = "GPL"

import h5py
import numpy as np


def del_bins(filename, N0, N):
    """Delete N bins in all observables of the specified HDF5-file.

    filename: Name of HDF5 file
    N0: Number of first N0 bins to leave
    N: Number of bins to remove after first N0 bins
    """
    def reshape(fileobj, dset_name, N0, N):
        dset = fileobj[dset_name]
        dat = np.copy(np.concatenate([dset[:N0], dset[N0+N:]]))
        fileobj[dset_name].resize(dat.shape)
        fileobj[dset_name][:] = dat

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


def show_obs(filename):
    """Show observables and their number of bins in ALF HDF5 file."""
    with h5py.File(filename, 'r') as f:
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


def bin_count(filename):
    """Count number of bins in ALF HDF5 file.

    Assumes all observables have same number of bins.
    """
    with h5py.File(filename, 'r') as f:
        N_bins = 0
        for o in f:
            if o.endswith('_scal') or o.endswith('_eq') \
               or o.endswith('_tau') or o.endswith('_hist'):
                N_bins = f[o+"/obser"].shape[0]
                break
        print(filename, N_bins)
