"""Analysis routines."""
# pylint: disable=invalid-name
# pylint: disable=too-many-branches
# pylint: disable=missing-function-docstring
# pylint: disable=unbalanced-tuple-unpacking

import os
import shutil
import pickle

import h5py
import numpy as np
import pandas as pd
import f90nml

from . lattice import Lattice
from . exceptions import TooFewBinsError


def symmetrize(latt, syms, dat):
    """
    Symmetrize a dataset.

    Parameters
    ----------
    latt : Lattice
        See :class:`py_alf.Lattice`.
    syms : list
        List of symmetry operations, including the identity of the form
        sym(latt, i) -> i_tranformed
    dat : array-like object
        Data to symmetrize. The symmetrization is with respect to the last
        index of dat.

    Returns
    -------
    dat_sym : numpy array
        Symmetrized data.
    """
    N = dat.shape[-1]
    N_sym = len(syms)
    dat_sym = np.zeros(dat.shape, dtype=dat.dtype)

    for i in range(N):
        for sym in syms:
            dat_sym[..., i] += dat[..., sym(latt, i)]/N_sym

    return dat_sym


class Parameters:
    """
    Object representing the "parameters" file.

    Parameters
    ----------
    directory : path-like object
        Directory of "parameters" file.
    obs_name : str, optional
        Observable name. If this is set, the object tries to get a parameters
        not from the namelist 'var_errors', but from a namelist called
        `obs_name`, while 'var_errors' is the fallback options. Parameters
        will be written to namelist `obs_name`.
    """

    def __init__(self, directory, obs_name=None):
        self.directory = directory
        self.filename = os.path.join(directory, 'parameters')
        self._nml = f90nml.read(self.filename)
        if obs_name is None:
            self.obs_name = 'var_errors'
        else:
            self.obs_name = obs_name.lower()

    def write_nml(self):
        """Write namelist to file. Preseves comments."""
        f90nml.patch(self.filename, self._nml, self.filename+'_temp')
        shutil.move(self.filename+'_temp', self.filename)

    def _get_parameter(self, parameter_name):
        try:
            return self._nml[self.obs_name][parameter_name]
        except KeyError:
            return self._nml['var_errors'][parameter_name]

    def N_skip(self):
        """Get N_skip."""
        return self._get_parameter('n_skip')

    def N_rebin(self):
        """Get N_rebin."""
        return self._get_parameter('n_rebin')

    def _set_parameter(self, parameter_name, parameter):
        try:
            temp = self._nml[self.obs_name]
        except KeyError:
            temp = {}

        temp[parameter_name] = parameter
        self._nml[self.obs_name] = temp

    def set_N_skip(self, parameter):
        """Update N_skip."""
        self._set_parameter('n_skip', parameter)

    def set_N_rebin(self, parameter):
        """Update N_rebin."""
        self._set_parameter('n_rebin', parameter)

    def N_min(self):
        """Get minimal number of bins, given the parameters in this object."""
        return self.N_skip() + 2*self.N_rebin()


def rebin(X, N_rebin):
    """Combine each N_rebin bins into one bin.

    If the number of bins (=N0) is not an integer multiple of N_rebin,
    the last N0 modulo N_rebin bins are discarded.
    """
    if N_rebin == 1:
        return X
    N0 = len(X)
    N = N0 // N_rebin
    shape = (N,) + X.shape[1:]
    Y = np.empty(shape, dtype=X.dtype)
    for i in range(N):
        Y[i] = np.mean(X[i*N_rebin:(i+1)*N_rebin], axis=0)
    return Y


def jack(X, par, N_skip=None, N_rebin=None):
    """
    Create jackknife bins out of input bins after skipping and rebinning.

    Parameters
    ----------
    X : array-like object
        Input bins. Bins run over first index.
    par : :class:`Parameters`
        Parameters object.
    N_skip : int, default=par.N_skip()
        Number of bins to skip.
    N_rebin : int, default=par.N_rebin()
        Number of bins to recombine into one.

    Returns
    -------
    numpy array
        Jackknife bins after skipping and rebinning.
    """
    if N_rebin is None:
        N_rebin = par.N_rebin()
    if N_skip is None:
        N_skip = par.N_skip() + (len(X)-par.N_skip()) % N_rebin
    if N_skip != 0:
        X = X[N_skip:]
    X = rebin(X, N_rebin)
    N = len(X)
    Y = (np.sum(X, axis=0) - X) / (N-1)
    return Y


def error(jacks, imag=False):
    """
    Calculate expectation values and errors of given jackknife bins.

    Parameters
    ----------
    jacks : array-like object
        Jackknife bins.
    imag : bool, default=False
        Output with imaginary part.

    Returns
    -------
    tuple of numpy arrays
        (expectation values, errors).
    """
    N = len(jacks)
    m_r = np.mean(jacks.real, axis=0)
    e_r = np.sqrt(np.var(jacks.real, axis=0) * N)
    if imag:
        m_i = np.mean(jacks.imag, axis=0)
        e_i = np.sqrt(np.var(jacks.imag, axis=0) * N)
        return m_r, e_r, m_i, e_i
    return m_r, e_r


def read_scal(directory, obs_name, bare_bins=False):
    """
    Read, skip, rebin and jackknife scalar-type bins.

    Bins get skipped and rebinned according to N_skip an N_rebin retrieved
    through :class:`Parameters`, then jackknife resampling is applied.

    Parameters
    ----------
    directory : path-like object
        Directory containing the observable.
    obs_name : str
        Name of the observable.
    bare_bins : bool, default=False
        Do not perform skipping, rebinning, or jackknife resampling.

    Returns
    -------
    array
        Observables. shape: `(N_bins, N_obs)`.
    array
        Sign. shape: `(N_bins,)`.
    N_obs : int
        Number of observables.
    """
    if 'data.h5' in os.listdir(directory):
        filename = os.path.join(directory, 'data.h5')

        with h5py.File(filename, "r") as f:       # pylint: disable=no-member
            obs = f[obs_name + "/obser"]  # Indices: bins, n_obs, re/im
            obs_c = obs[..., 0] + 1j * obs[..., 1]
            N_obs = obs_c.shape[1]

            sign = np.copy(f[obs_name + "/sign"])  # Indices: bins
    else:
        filename = os.path.join(directory, obs_name)

        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()

        N_bins = len(lines)
        N_obs = int(lines[0].split()[0])-1

        obs_c = np.empty([N_bins, N_obs], dtype=complex)
        sign = np.empty([N_bins], dtype=float)

        for i in range(N_bins):
            obs_c[i] = np.loadtxt(
                lines[i].replace(',', '+').replace(')', 'j)').split()[1:-1],
                dtype=complex)
            sign[i] = float(lines[i].split()[-1])

    if bare_bins:
        return obs_c, sign, N_obs

    par = Parameters(directory, obs_name)
    J_sign = jack(sign, par)
    J_obs = jack(obs_c, par)
    N_obs = J_obs.shape[1]
    return J_obs, J_sign, N_obs


def read_hist(directory, obs_name, bare_bins=False):
    """
    Read, skip, rebin and jackknife histogram-type bins.

    Bins get skipped and rebinned according to N_skip an N_rebin retrieved
    through :class:`Parameters`, then jackknife resampling is applied.

    Parameters
    ----------
    directory : path-like object
        Directory containing the observable.
    obs_name : str
        Name of the observable.
    bare_bins : bool, default=False
        Do not perform skipping, rebinning, or jackknife resampling.

    Returns
    -------
    array
        Observables. shape: `(N_bins, N_classes)`.
    array
        Sign. shape: `(N_bins,)`.
    array
        Proportion of observations above upper bound. shape: `(N_bins,)`.
    array
        Proportion of observations below lower bound. shape: `(N_bins,)`.
    N_classes : int
        Number of classes between upper and lower bound.
    upper : float
        Upper bound.
    lower : float
        Lower bound.
    """
    # pylint: disable=too-many-locals
    par = Parameters(directory, obs_name)

    if 'data.h5' in os.listdir(directory):
        filename = os.path.join(directory, 'data.h5')

        with h5py.File(filename, "r") as f:       # pylint: disable=no-member
            obs = f[obs_name + "/obser"]  # Indices: bins, n_classes
            sign = f[obs_name + "/sign"]  # Indices: bins
            above = f[obs_name + "/above"]  # Indices: bins
            below = f[obs_name + "/below"]  # Indices: bins
            N_classes = f[obs_name].attrs['N_classes']
            upper = f[obs_name].attrs['upper']
            lower = f[obs_name].attrs['lower']

            if bare_bins:
                return (np.copy(obs), np.copy(sign), np.copy(above),
                        np.copy(below), N_classes, upper, lower)

            J_obs = jack(obs, par)
            J_sign = jack(sign, par)
            J_above = jack(above, par)
            J_below = jack(below, par)
    else:
        filename = os.path.join(directory, obs_name)

        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()

        N_bins = len(lines)

        N_classes = int(lines[0].split()[0])
        upper = float(lines[0].split()[1])
        lower = float(lines[0].split()[2])

        above = np.empty([N_bins], dtype='float_')
        below = np.empty([N_bins], dtype='float_')
        obs = np.empty([N_bins, N_classes], dtype='float_')
        sign = np.empty([N_bins], dtype='float_')

        for i in range(N_bins):
            above[i] = float(lines[i].split()[3])
            below[i] = float(lines[i].split()[4])
            obs[i] = np.loadtxt(
                lines[i].replace(',', '+').replace(')', 'j)').split()[5:-1])
            sign[i] = float(lines[i].split()[-1])

        if bare_bins:
            return obs, sign, above, below, N_classes, upper, lower

        J_obs = jack(obs, par)
        J_sign = jack(sign, par)
        J_above = jack(above, par)
        J_below = jack(below, par)

    return J_obs, J_sign, J_above, J_below, N_classes, upper, lower


def read_latt(directory, obs_name, bare_bins=False, substract_back=True):
    """
    Read, skip, rebin and jackknife lattice-type bins (_eq and _tau).

    Bins get skipped and rebinned according to N_skip an N_rebin retrieved
    through :class:`Parameters`, then jackknife resampling is applied.

    Parameters
    ----------
    directory : path-like object
        Directory containing the observable.
    obs_name : str
        Name of the observable.
    bare_bins : bool, default=False
        Do not perform skipping, rebinning, or jackknife resampling.
    substract_back : bool, default=True
        Substract background from correlation functions.

    Returns
    -------
    array
        Observables. shape: `(N_bins, N_orb, N_orb, N_tau, latt.N)`.
    array
        Background. shape: `(N_bins, N_orb)`
    array
        Sign. shape: `(N_bins,)`.
    N_orb : int
        Number of orbitals.
    N_tau : int
        Number of imaginary time steps.
    dtau : float
        Imaginary time step length.
    latt : Lattice
        See :class:`py_alf.Lattice`.
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-nested-blocks
    par = Parameters(directory, obs_name)
    filename = os.path.join(directory, 'data.h5')

    if 'data.h5' in os.listdir(directory):
        filename = os.path.join(directory, 'data.h5')
        with h5py.File(filename, "r") as f:       # pylint: disable=no-member
            latt = Lattice(f[obs_name]["lattice"].attrs)

            obs = f[obs_name + "/obser"]
            # Indices: bins, no1, no, nt, n, re/im
            obs_c = obs[..., 0] + 1j * obs[..., 1]

            back = f[obs_name + "/back"]  # Indices: bins, no, re/im
            back_c = back[..., 0] + 1j * back[..., 1]

            sign = np.copy(f[obs_name + "/sign"])  # Indices: bins

            N_orb = obs.shape[1]
            N_tau = obs.shape[3]
            dtau = f[obs_name].attrs['dtau']
    else:
        filename = os.path.join(directory, obs_name)
        with open(filename+'_info', 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        # Channel = lines[1].split(':')[1].strip()
        N_tau = int(lines[2].split(':')[1])
        dtau = float(lines[3].split(':')[1])
        L1_p = np.fromstring(lines[6].split(':')[1], sep=' ')
        L2_p = np.fromstring(lines[7].split(':')[1], sep=' ')
        a1_p = np.fromstring(lines[8].split(':')[1], sep=' ')
        a2_p = np.fromstring(lines[9].split(':')[1], sep=' ')
        N_orb = int(lines[12].split(':')[1])

        latt = Lattice(L1_p, L2_p, a1_p, a2_p)
        N_unit = latt.N

        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()

        N_bins0 = len(lines) / (1 + N_orb + N_unit + N_unit*N_tau*N_orb**2)
        N_bins = int(round(N_bins0))
        if N_bins0 - N_bins > 1e-10:
            raise RuntimeError(
                f'Error in read_latt_plaintxt: File "{filename}" '
                'lines number does not fit.')

        latt = Lattice(L1_p, L2_p, a1_p, a2_p)

        obs_c = np.empty((N_bins, N_orb, N_orb, N_tau, N_unit), dtype=complex)
        back_c = np.empty((N_bins, N_orb), dtype=complex)
        sign = np.empty((N_bins,), dtype=float)

        i_line = 0
        for i_bin in range(N_bins):
            sign[i_bin] = float(lines[i_line].split()[0])
            i_line += 1
            for i_orb in range(N_orb):
                back_c[i_bin, i_orb] = complex(
                    lines[i_line].replace(',', '+').replace(')', 'j)'))
                i_line += 1
            for i_unit in range(N_unit):
                i_line += 1
                for i_tau in range(N_tau):
                    for i_orb in range(N_orb):
                        for i_orb1 in range(N_orb):
                            obs_c[i_bin, i_orb1, i_orb, i_tau, i_unit] = \
                                complex(lines[i_line]
                                        .replace(',', '+')
                                        .replace('+-', '-')
                                        .replace(')', 'j)'))
                            i_line += 1

    if bare_bins:
        if substract_back:
            # Substract background
            n = latt.invlistk[0, 0]
            for no in range(N_orb):
                for no1 in range(N_orb):
                    for nt in range(N_tau):
                        obs_c[:, no1, no, nt, n] -= \
                            latt.N*back_c[:, no1]*back_c[:, no]
        return obs_c, back_c, sign, N_orb, N_tau, dtau, latt
    J_obs = jack(obs_c, par)
    J_back = jack(back_c, par)
    J_sign = jack(sign, par)

    if substract_back:
        # Substract background
        n = latt.invlistk[0, 0]
        for no in range(N_orb):
            for no1 in range(N_orb):
                for nt in range(N_tau):
                    J_obs[:, no1, no, nt, n] \
                        -= latt.N*J_back[:, no1]*J_back[:, no]
    return J_obs, J_back, J_sign, N_orb, N_tau, dtau, latt


class ReadObs:
    """
    Read, skip, rebin and jackknife scalar-type bins.

    Bins get skipped and rebinned according to N_skip an N_rebin retrieved
    through :class:`Parameters`, then jackknife resampling is applied.
    Saves jackknife bins.

    Cf. :func:`read_scal`, :func:`read_latt`, :func:`read_hist`.

    Parameters
    ----------
    directory : path-like object
        Directory containing the observable.
    obs_name : str
        Name of observable.
    bare_bins : bool, default=False
        Do not perform skipping, rebinning, or jackknife resampling.
    substract_back : bool, default=True
        Substract background. Applies to correlation functions.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, directory, obs_name,
                 bare_bins=False, substract_back=True):
        self.directory = directory
        self.obs_name = obs_name
        self.bare_bins = bare_bins
        if obs_name.endswith('_scal'):
            self.J_obs, self.J_sign, self.N_obs = \
                read_scal(directory, obs_name, bare_bins)
        elif obs_name.endswith('_eq') or obs_name.endswith('_tau'):
            (self.J_obs, self.J_back, self.J_sign, self.N_orb, self.N_tau,
             self.dtau, self.latt) = \
                read_latt(directory, obs_name, bare_bins, substract_back)
        elif obs_name.endswith('_hist'):
            (self.J_obs, self.J_sign, self.J_above, self.J_below,
             self.N_classes, self.upper, self.lower) = \
                read_hist(directory, obs_name, bare_bins)
        else:
            raise TypeError('Error in ReadObs.init: Unknow observable type.')
        self.N_bins = self.J_obs.shape[0]

    def all(self):
        """Return all bins."""
        if self.obs_name.endswith('_scal'):
            return self.J_obs, self.J_sign, self.N_obs
        if self.obs_name.endswith('_eq') or self.obs_name.endswith('_tau'):
            return (self.J_obs, self.J_back, self.J_sign, self.N_orb,
                    self.N_tau, self.dtau, self.latt)
        if self.obs_name.endswith('_hist'):
            return (self.J_obs, self.J_sign, self.J_above, self.J_below,
                    self.N_classes, self.upper, self.lower)
        raise TypeError('Error in ReadObs.all: Unknow observable type.')

    def slice(self, n):
        """Return n-th bin."""
        if self.obs_name.endswith('_scal'):
            return self.J_obs[n], self.J_sign[n], self.N_obs
        if self.obs_name.endswith('_eq') or self.obs_name.endswith('_tau'):
            return (self.J_obs[n], self.J_back[n], self.J_sign[n], self.N_orb,
                    self.N_tau, self.dtau, self.latt)
        if self.obs_name.endswith('_hist'):
            return (self.J_obs[n], self.J_sign[n], self.J_above[n],
                    self.J_below[n], self.N_classes, self.upper, self.lower)
        raise TypeError('Error in ReadObs.slice: Unknow observable type.')

    def jack(self, N_rebin):
        """
        Return jackknife bins. Object has to be created with `bare_bins=True`.

        Parameters
        ----------
        N_rebin : int
            Overwrite N_rebin from parameters.
        """
        if not self.bare_bins:
            raise TypeError('Object has to be created with `bare_bins=True`.')
        par = Parameters(self.directory)
        J_obs_temp = jack(self.J_obs, par, N_rebin=N_rebin)
        N = len(J_obs_temp)
        if self.obs_name.endswith('_scal'):
            return (J_obs_temp,
                    jack(self.J_sign, par, N_rebin=N_rebin),
                    N*[self.N_obs])
        if self.obs_name.endswith('_eq') or self.obs_name.endswith('_tau'):
            return (J_obs_temp,
                    jack(self.J_back, par, N_rebin=N_rebin),
                    jack(self.J_sign, par, N_rebin=N_rebin),
                    N*[self.N_orb], N*[self.N_tau], N*[self.dtau],
                    N*[self.latt])
        if self.obs_name.endswith('_hist'):
            return (J_obs_temp,
                    jack(self.J_sign, par, N_rebin=N_rebin),
                    jack(self.J_above, par, N_rebin=N_rebin),
                    jack(self.J_below, par, N_rebin=N_rebin),
                    N*[self.N_classes], N*[self.upper], N*[self.lower])
        raise TypeError('Error in ReadObs.jack: Unknow observable type.')


def ana_scal(directory, obs_name):
    """Analyze given scalar observables.

    Parameters
    ----------
    directory : path-like object
        Directory containing the observable.
    obs_name : str
        Name of the observable.
    """
    J_obs, J_sign, N_obs = ReadObs(directory, obs_name).all()
    if len(J_obs) < 2:
        raise TooFewBinsError()

    sign = error(J_sign)

    dat = np.empty((N_obs, 2))
    for n in range(N_obs):
        J = J_obs[:, n] / J_sign
        dat[n, :] = error(J)

    return sign, dat


def ana_hist(directory, obs_name):
    """Analyze given histogram observables."""
    # pylint: disable=too-many-locals
    J_obs, J_sign, J_above, J_below, N_classes, upper, lower = \
        ReadObs(directory, obs_name).all()
    if len(J_obs) < 2:
        raise TooFewBinsError()

    sign = error(J_sign)
    above = error(J_above)
    below = error(J_below)

    d_class = (upper-lower)/N_classes
    dat = np.empty((N_classes, 3))
    for n in range(N_classes):
        J = J_obs[:, n] / J_sign
        dat[n, :] = [lower+d_class*(0.5+n), *error(J)]

    return sign, above, below, dat, upper, lower


def ana_eq(directory, obs_name, sym=None):
    """Analyze given equal-time correlators.

    If sym is given, it symmetrizes the bins prior to calculating the error.
    Cf. :func:`symmetrize`.
    """
    # pylint: disable=too-many-locals
    J_obs, J_back, J_sign, N_orb, N_tau, dtau, latt = \
        ReadObs(directory, obs_name).all()
    del J_back, N_tau, dtau
    N_bins = len(J_sign)
    if N_bins < 2:
        raise TooFewBinsError()

    J_obs = J_obs.reshape((N_bins, N_orb, N_orb, latt.N))

    m, e = error(J_sign)
    sign = (m, e)

    J = np.array([J_obs[n] / J_sign[n] for n in range(N_bins)])

    if sym is not None:
        J = symmetrize(latt, sym, J)

    m_K, e_K = error(J)

    J_sum = J.trace(axis1=1, axis2=2)
    m_sum, e_sum = error(J_sum)

    J_R = latt.fourier_K_to_R(J)
    m_R, e_R = error(J_R)

    J_R_sum = latt.fourier_K_to_R(J_sum)
    m_R_sum, e_R_sum = error(J_R_sum)

    return sign, m_K, e_K, m_sum, e_sum, m_R, e_R, m_R_sum, e_R_sum, latt


def ana_tau(directory, obs_name, sym=None):
    """Analyze given time-displaced correlators.

    If sym is given, it symmetrizes the bins prior to calculating the error.
    Cf. :func:`symmetrize`.
    """
    # pylint: disable=too-many-locals
    J_obs, J_back, J_sign, N_orb, N_tau, dtau, latt = \
        ReadObs(directory, obs_name).all()
    if len(J_obs) < 2:
        raise TooFewBinsError()
    del J_back, N_orb, N_tau
    N_bins = len(J_sign)

    m, e = error(J_sign)
    sign = (m, e)

    J = np.array(
        [J_obs[n].trace(axis1=0, axis2=1) / J_sign[n] for n in range(N_bins)])

    if sym is not None:
        J = symmetrize(latt, sym, J)

    m_K, e_K = error(J)

    # Fourier transform
    J_R = latt.fourier_K_to_R(J)
    m_R, e_R = error(J_R)

    return sign, m_K, e_K, m_R, e_R, dtau, latt


def write_res_eq(directory, obs_name,
                 m_k, e_k, m_k_sum, e_k_sum,
                 m_r, e_r, m_r_sum, e_r_sum, latt):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    N_orb = m_k.shape[0]
    header = ['kx', 'ky']
    out = latt.k
    fmt = '\t'.join(['%8.5f %8.5f'] + ['% 13.10e % 13.10e']*(N_orb**2+1))
    fmth = '\t'.join(['{:^8s} {:^8s}'] + [' {:^33s}']*(N_orb**2+1))
    for no in range(N_orb):
        for no1 in range(N_orb):
            header = header + [str((no, no1))]
            out = np.column_stack([out, m_k[no1, no], e_k[no1, no]])
    out = np.column_stack([out, m_k_sum, e_k_sum])
    header = header + ['trace over n_orb']

    np.savetxt(
        os.path.join(directory, 'res', obs_name + '_K'),
        out,
        fmt=fmt,
        header=fmth.format(*header)
        )

    header = ['kx', 'ky']
    out = latt.k
    fmt = '\t'.join(['%8.5f %8.5f'] + ['% 13.10e % 13.10e']*(1))
    fmth = '\t'.join(['{:^8s} {:^8s}'] + [' {:^33s}']*(1))
    out = np.column_stack([out, m_k_sum, e_k_sum])
    header = header + ['trace over n_orb']
    np.savetxt(
        os.path.join(directory, 'res', obs_name + '_K_sum'),
        out,
        fmt=fmt,
        header=fmth.format(*header)
        )

    header = ['rx', 'ry']
    out = latt.r
    fmt = '\t'.join(['%9.5f %9.5f'] + ['% 13.10e % 13.10e']*(N_orb**2+1))
    fmth = '\t'.join(['{:^8s} {:^8s}'] + [' {:^33s}']*(N_orb**2+1))
    for no in range(N_orb):
        for no1 in range(N_orb):
            header = header + [str((no, no1))]
            out = np.column_stack([out, m_r[no1, no], e_r[no1, no]])
    out = np.column_stack([out, m_r_sum, e_r_sum])
    header = header + ['trace over n_orb']

    np.savetxt(
        os.path.join(directory, 'res', obs_name + '_R'),
        out,
        fmt=fmt,
        header=fmth.format(*header)
        )

    header = ['rx', 'ry']
    out = latt.r
    fmt = '\t'.join(['%9.5f %9.5f'] + ['% 13.10e % 13.10e']*(1))
    fmth = '\t'.join(['{:^8s} {:^8s}'] + [' {:^33s}']*(1))
    out = np.column_stack([out, m_r_sum, e_r_sum])
    header = header + ['trace over n_orb']
    np.savetxt(
        os.path.join(directory, 'res', obs_name + '_R_sum'),
        out,
        fmt=fmt,
        header=fmth.format(*header)
        )


def write_res_tau(directory, obs_name, m_k, e_k, m_r, e_r, dtau, latt):
    # pylint: disable=too-many-arguments
    N_tau = m_k.shape[0]
    taus = np.linspace(0., (N_tau-1)*dtau, num=N_tau)

    for n in range(latt.N):
        directory2 = os.path.join(
            directory, 'res', obs_name, '{0:.2f}_{1:.2f}'.format(*latt.k[n]))
        if not os.path.exists(directory2):
            os.makedirs(directory2)

        np.savetxt(os.path.join(directory2, 'dat'),
                   np.column_stack([taus, m_k[:, n], e_k[:, n]]),
                   fmt=['%14.7f', '%16.8f', '%16.8f']
                   )

    n = latt.invlistr[0, 0]
    np.savetxt(os.path.join(directory, 'res', obs_name, 'R0'),
               np.column_stack([taus, m_r[:, n], e_r[:, n]]),
               fmt=['%14.7f', '%16.8f', '%16.8f']
               )


def load_res(directories):
    """
    Read analysis results from multiple simulations.

    Read from pickled dictionaries 'res.pkl' and return everything
    in a single pandas DataFrame with one row per simulation.

    Parameters
    ----------
    directories : list of path-like objects
        Directories containing analyzed simulation results.

    Returns
    -------
    df : pandas.DataFrame
        Contains analysis results and Hamiltonian parameters.
        One row per simulation.
    """
    if not isinstance(directories, list):
        directories = [directories]
    li = []
    directories_in = []
    for directory in directories:
        print(directory)
        try:
            with open(os.path.join(directory, 'res.pkl'), 'rb') as f:
                dictionary = pickle.load(f)
        except FileNotFoundError:
            print(f"No file named 'res.pkl' in {directory}, skipping.")
            continue
        directories_in.append(directory)
        with h5py.File(os.path.join(directory, 'data.h5'), "r") as f:
            dictionary.update(f['parameters'].attrs)

            dictionary['lattice'] = {}
            dictionary['lattice'].update(f["lattice"].attrs)
            try:
                dictionary['lattice']['orbitals'] = np.copy(f["lattice/orbitals"])
            except KeyError:
                print('No orbital locations saved.')
        li.append(dictionary)

    df = pd.DataFrame(li, index=directories_in)
    return df


def custom_obs_get_dtype_len(obs_spec, bins):
    sample = obs_spec['function'](
                    *[x for b in bins for x in b.slice(0)],
                    **obs_spec['kwargs'])
    dypte = sample.dtype
    if np.ndim(sample) == 0:
        length = 1
    elif np.ndim(sample) == 1:
        length = len(sample)
    else:
        raise TypeError(
            "Only scalar or one-dimensional custom_obs allowed")
    return dypte, length
