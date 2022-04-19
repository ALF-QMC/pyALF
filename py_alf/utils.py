"""Utility functions for handling ALF HDF5 files."""
# pylint: disable=invalid-name
# py lint: disable=consider-using-f-string

__author__ = "Jonas Schwab"
__copyright__ = "Copyright 2022, The ALF Project"
__license__ = "GPL"

import os

import h5py
import numpy as np


def find_sim_dirs(root_in='.'):
    """
    Find directories containing a file named 'data.h5'.

    Parameters
    ----------
    root_in : path-like object, default='.'
        Root directory from where to start searching.

    Returns
    -------
    list of directory names.
    """
    dirs = []
    for root, folders, files in os.walk(root_in):
        del folders
        if 'data.h5' in files:
            dirs.append(root)
    dirs.sort()
    return dirs


def del_bins(filename, N0, N):
    """Delete N bins in all observables of the specified HDF5-file.

    Parameters
    ----------
    filename: str
        Name of HDF5 file.
    N0: int
        Number of first N0 bins to keep.
    N: int
        Number of bins to remove after first N0 bins.
    """
    def reshape(fileobj, dset_name, N0, N):
        dset = fileobj[dset_name]
        dat = np.copy(np.concatenate([dset[:N0], dset[N0+N:]]))
        fileobj[dset_name].resize(dat.shape)
        fileobj[dset_name][:] = dat

    with h5py.File(filename, 'r+') as f:          # pylint: disable=no-member
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
    """Show observables and their number of bins in the given ALF HDF5 file.

    Parameters
    ----------
    filename: str
        Name of HDF5 file.
    """
    with h5py.File(filename, 'r') as f:           # pylint: disable=no-member
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
    """Count number of bins in the given ALF HDF5 file.

    Assumes all observables have the same number of bins.

    Parameters
    ----------
    filename: str
        Name of HDF5 file.
    """
    with h5py.File(filename, 'r') as f:           # pylint: disable=no-member
        N_bins = 0
        for o in f:
            if o.endswith('_scal') or o.endswith('_eq') \
               or o.endswith('_tau') or o.endswith('_hist'):
                N_bins = f[o+"/obser"].shape[0]
                break
        print(filename, N_bins)
