# pylint: disable=invalid-name
# pylint: disable=too-many-branches
# pylint: disable=missing-function-docstring

import os
import pickle

import h5py
import numpy as np

from alf_ana.lattice import Lattice


def symmetrize(latt, syms, dat):
    """Symmetrizes a dataset, where syms is the list of symmetry operations,
    including the identity, and dat is the data. The symmetrization is with
    respect to the last index of dat
    """
    N = dat.shape[-1]
    N_sym = len(syms)
    dat_sym = np.zeros(dat.shape, dtype=dat.dtype)

    for i in range(N):
        for sym in syms:
            dat_sym[..., i] += dat[..., sym(latt, i)]/N_sym

    return dat_sym


class Parameters:
    """Object representing the "parameters" file """

    def __init__(self, directory, obs_name=None):
        try:
            import f90nml
            self.directory = directory
            self.filename = os.path.join(directory, 'parameters')
            self._nml = f90nml.read(self.filename)
            if obs_name is None:
                self.obs_name = 'var_errors'
            else:
                self.obs_name = obs_name.lower()
        except ImportError:
            Exception(
                'Loading of f90nml failed, no reading of parameters file.')

    def write_nml(self):
        self._nml.write(self.filename, force=True)

    def get_parameter(self, parameter_name):
        try:
            return self._nml[self.obs_name][parameter_name]
        except Exception:
            return self._nml['var_errors'][parameter_name]

    def N_skip(self):
        return self.get_parameter('n_skip')

    def N_rebin(self):
        return self.get_parameter('n_rebin')

    def set_parameter(self, parameter_name, parameter):
        try:
            temp = self._nml[self.obs_name]
        except Exception:
            temp = {}

        temp[parameter_name] = parameter
        self._nml[self.obs_name] = temp

    def set_N_skip(self, parameter):
        self.set_parameter('n_skip', parameter)

    def set_N_rebin(self, parameter):
        self.set_parameter('n_rebin', parameter)

    def N_min(self):
        return self.N_skip() + 2*self.N_rebin()


def rebin(X, N_rebin):
    '''Combines each N_rebin bins into one bin. If the number of bins (=N0)
    is not an integer multiple of N_rebin, the last N0 modulo N_rebin bins get
    dismissed.
    '''
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
    '''Creates jackknife bins out of input bins after after skipping and
    rebinning according to par. N_rebin overwrites N_rebin of par.
    '''
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
    '''Calculates expectation values and erros of given jackknife bins.'''
    N = len(jacks)
    m_r = np.mean(jacks.real, axis=0)
    e_r = np.sqrt(np.var(jacks.real, axis=0) * N)
    if imag:
        m_i = np.mean(jacks.imag, axis=0)
        e_i = np.sqrt(np.var(jacks.imag, axis=0) * N)
        return m_r, e_r, m_i, e_i
    else:
        return m_r, e_r


class ReadObs:
    '''Reads in bins of arbitraty format and performs skipping of bins and
    rebinning as specified in parameters file. Returns jackknife bins'''

    def __init__(self, directory, obs_name,
                 bare_bins=False, substract_back=True):
        self.directory = directory
        self.obs_name = obs_name
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
            raise Exception('Error in ReadObs.init')
        self.N_bins = self.J_obs.shape[0]

    def all(self):
        if self.obs_name.endswith('_scal'):
            return self.J_obs, self.J_sign, self.N_obs
        elif self.obs_name.endswith('_eq') or self.obs_name.endswith('_tau'):
            return (self.J_obs, self.J_back, self.J_sign, self.N_orb,
                    self.N_tau, self.dtau, self.latt)
        elif self.obs_name.endswith('_hist'):
            return (self.J_obs, self.J_sign, self.J_above, self.J_below,
                    self.N_classes, self.upper, self.lower)
        else:
            raise Exception('Error in ReadObs.all')

    def slice(self, n):
        if self.obs_name.endswith('_scal'):
            return self.J_obs[n], self.J_sign[n], self.N_obs
        elif self.obs_name.endswith('_eq') or self.obs_name.endswith('_tau'):
            return (self.J_obs[n], self.J_back[n], self.J_sign[n], self.N_orb,
                    self.N_tau, self.dtau, self.latt)
        elif self.obs_name.endswith('_hist'):
            return (self.J_obs[n], self.J_sign[n], self.J_above[n],
                    self.J_below[n], self.N_classes, self.upper, self.lower)
        else:
            raise Exception('Error in ReadObs.slice')

    def jack(self, N_rebin):
        par = Parameters(self.directory)
        J_obs_temp = jack(self.J_obs, par, N_rebin=N_rebin)
        N = len(J_obs_temp)
        if self.obs_name.endswith('_scal'):
            return (J_obs_temp,
                    jack(self.J_sign, par, N_rebin=N_rebin),
                    N*[self.N_obs])
        elif self.obs_name.endswith('_eq') or self.obs_name.endswith('_tau'):
            return (J_obs_temp,
                    jack(self.J_back, par, N_rebin=N_rebin),
                    jack(self.J_sign, par, N_rebin=N_rebin),
                    N*[self.N_orb], N*[self.N_tau], N*[self.dtau], N*[self.latt])
        elif self.obs_name.endswith('_hist'):
            return (J_obs_temp,
                    jack(self.J_sign, par, N_rebin=N_rebin),
                    jack(self.J_above, par, N_rebin=N_rebin),
                    jack(self.J_below, par, N_rebin=N_rebin),
                    N*[self.N_classes], N*[self.upper], N*[self.lower])
        else:
            raise Exception('Error in ReadObs.jack')


def read_scal(directory, obs_name, bare_bins=False):
    '''Reads in scalar-type bins and performs skipping of bins and rebinning
    as specified in parameters file. Returns jackknife bins'''
    if 'data.h5' in os.listdir(directory):
        filename = os.path.join(directory, 'data.h5')

        with h5py.File(filename, "r") as f:
            obs = f[obs_name + "/obser"]  # Indices: bins, n_obs, re/im
            obs_c = obs[..., 0] + 1j * obs[..., 1]
            N_obs = obs_c.shape[1]

            sign = np.copy(f[obs_name + "/sign"])  # Indices: bins
    else:
        filename = os.path.join(directory, obs_name)

        with open(filename, 'r') as f:
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
    '''Reads in histogram-type bins and performs skipping of bins and rebinning
    as specified in parameters file. Returns jackknife bins'''
    par = Parameters(directory, obs_name)

    if 'data.h5' in os.listdir(directory):
        filename = os.path.join(directory, 'data.h5')

        with h5py.File(filename, "r") as f:
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

        with open(filename, 'r') as f:
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
    '''Reads in Lattice-type bins and performs skipping of bins and rebinning
    as specified in parameters file. Returns jackknife bins'''
    par = Parameters(directory, obs_name)
    filename = os.path.join(directory, 'data.h5')

    if 'data.h5' in os.listdir(directory):
        filename = os.path.join(directory, 'data.h5')
        with h5py.File(filename, "r") as f:
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
        with open(filename+'_info', 'r') as f:
            lines = f.readlines()
        Channel = lines[1].split(':')[1].strip()
        N_tau = int(lines[2].split(':')[1])
        dtau = float(lines[3].split(':')[1])
        L1_p = np.fromstring(lines[6].split(':')[1], sep=' ')
        L2_p = np.fromstring(lines[7].split(':')[1], sep=' ')
        a1_p = np.fromstring(lines[8].split(':')[1], sep=' ')
        a2_p = np.fromstring(lines[9].split(':')[1], sep=' ')
        N_orb = int(lines[12].split(':')[1])

        latt = Lattice(L1_p, L2_p, a1_p, a2_p)
        N_unit = latt.N

        with open(filename, 'r') as f:
            lines = f.readlines()

        N_bins0 = len(lines) / (1 + N_orb + N_unit + N_unit*N_tau*N_orb**2)
        N_bins = int(round(N_bins0))
        if N_bins0 - N_bins > 1e-10:
            raise Exception('Error in read_latt_plaintxt: File "{}" \
                            lines number does not fit'.format(filename))

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
                                    .replace(',', '+').replace('+-', '-').replace(')', 'j)'))
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
                    J_obs[:, no1, no, nt, n] -= latt.N*J_back[:, no1]*J_back[:, no]
    return J_obs, J_back, J_sign, N_orb, N_tau, dtau, latt


def ana_scal(filename, obs_name=None):
    '''Analyzes given scalar observables'''
    J_obs, J_sign, N_obs = ReadObs(filename, obs_name).all()

    sign = error(J_sign)

    dat = np.empty((N_obs, 2))
    for n in range(N_obs):
        J = J_obs[:, n] / J_sign
        dat[n, :] = error(J)

    return sign, dat


def ana_hist(filename, obs_name=None):
    '''Analyzes given histogram observables'''
    J_obs, J_sign, J_above, J_below, N_classes, upper, lower = \
        ReadObs(filename, obs_name).all()

    sign = error(J_sign)
    above = error(J_above)
    below = error(J_below)

    d_class = (upper-lower)/N_classes
    dat = np.empty((N_classes, 3))
    for n in range(N_classes):
        J = J_obs[:, n] / J_sign
        dat[n, :] = [lower+d_class*(0.5+n), *error(J)]

    return sign, above, below, dat, upper, lower


def ana_eq(filename, obs_name=None, sym=None):
    '''Analyzes given equal-time collalators.
    If sym is given, it symmetrizes the bins prior to calculating the error.
    '''
    J_obs, J_back, J_sign, N_orb, N_tau, dtau, latt = \
        ReadObs(filename, obs_name).all()
    del J_back, N_tau, dtau
    N_bins = len(J_sign)

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


def ana_tau(filename, obs_name=None, sym=None):
    '''Analyzes given timedisplaced corralators.
    If sym is given, it symmetrizes the bins prior to calculating the error.
    '''
    J_obs, J_back, J_sign, N_orb, N_tau, dtau, latt = \
        ReadObs(filename, obs_name).all()
    del J_back, N_orb, N_tau
    N_bins = len(J_sign)

    m, e = error(J_sign)
    sign = (m, e)

    J = np.array(
        [J_obs[n].trace(axis1=0, axis2=1) / J_sign[n] for n in range(N_bins)])

    if sym is not None:
        J = symmetrize(latt, sym, J)

    m_K, e_K = error(J)

    # Fourier transform, r=0
    J_R0 = J.sum(axis=2) / latt.N
    m_R0, e_R0 = error(J_R0)

    return sign, m_K, e_K, m_R0, e_R0, dtau, latt


def ana(directory, sym_spec=None, custom_obs=None, do_tau=True):
    """
    Performs analysis in given directory.
    """
    print('### Analyzing {} ###'.format(directory))
    print(os.getcwd())

    par = Parameters(directory)
    if 'data.h5' in os.listdir(directory):
        try:
            d1 = os.path.getmtime(os.path.join(directory, 'data.h5')) \
                - os.path.getmtime(os.path.join(directory, 'res.pkl'))
            d2 = os.path.getmtime(os.path.join(directory, 'parameters')) \
                - os.path.getmtime(os.path.join(directory, 'res.pkl'))
            if d1 < 0 and d2 < 0:
                print('already analyzed')
                return
        except OSError:
            pass

        with h5py.File(os.path.join(directory, 'data.h5'), "r") as f:
            list_obs = []
            list_scal = []
            list_hist = []
            list_eq = []
            list_tau = []

            N_bins = 0
            for o in f:
                if o.endswith('_scal'):
                    list_obs.append(o)
                    list_scal.append(o)
                    N_bins = max([N_bins, f[o + "/sign"].shape[0]])
                elif o.endswith('_hist'):
                    list_obs.append(o)
                    list_hist.append(o)
                    N_bins = max([N_bins, f[o + "/sign"].shape[0]])
                elif o.endswith('_eq'):
                    list_obs.append(o)
                    list_eq.append(o)
                    N_bins = max([N_bins, f[o + "/sign"].shape[0]])
                elif o.endswith('_tau'):
                    list_obs.append(o)
                    list_tau.append(o)
                    N_bins = max([N_bins, f[o + "/sign"].shape[0]])

        if N_bins < par.N_min():
            print('too few bins ', N_bins)
            return
    else:
        list_obs = []
        list_scal = []
        list_hist = []
        list_eq = []
        list_tau = []
        for o in os.listdir(directory):
            if o.endswith('_scal'):
                list_obs.append(o)
                list_scal.append(o)
            elif o.endswith('_hist'):
                list_obs.append(o)
                list_hist.append(o)
            elif o.endswith('_eq'):
                list_obs.append(o)
                list_eq.append(o)
            elif o.endswith('_tau'):
                list_obs.append(o)
                list_tau.append(o)

    if 'res' not in os.listdir(directory):
        os.mkdir(os.path.join(directory, 'res'))

    dic = {}

    if custom_obs is not None:
        print("Custom observables:")
        for obs_name in custom_obs:
            func = custom_obs[obs_name][0]
            o_in = custom_obs[obs_name][1]
            kwarg = custom_obs[obs_name][2]
            if all(x in list_obs for x in o_in):
                print('custom', obs_name, o_in)
                jacks = [ReadObs(directory, obs_name) for obs_name in o_in]

                N_bins = jacks[0].N_bins
                dtype = func(*[x for j in jacks for x in j.slice(0)],
                             **kwarg).dtype
                J = np.empty(N_bins, dtype=dtype)
                for i in range(N_bins):
                    J[i] = custom_obs[obs_name][0](
                        *[x for j in jacks for x in j.slice(i)], **kwarg)

                dat = error(J)

                dic[obs_name] = dat[0]
                dic[obs_name+'_err'] = dat[1]

                np.savetxt(
                    os.path.join(directory, 'res', obs_name),
                    dat
                    )

    print("Scalar observables:")
    for obs_name in list_scal:
        print(obs_name)
        sign, dat = ana_scal(directory, obs_name)

        dic[obs_name+'_sign'] = sign[0]
        dic[obs_name+'_sign_err'] = sign[1]
        for i in range(len(dat)):
            dic[obs_name+str(i)] = dat[i, 0]
            dic[obs_name+str(i)+'_err'] = dat[i, 1]

        np.savetxt(
            os.path.join(directory, 'res', obs_name),
            dat,
            header='Sign: {} {}'.format(*sign)
            )

    print("Histogram observables:")
    for obs_name in list_hist:
        print(obs_name)
        sign, above, below, dat, upper, lower = ana_hist(directory, obs_name)

        hist = {}
        hist['dat'] = dat
        hist['sign'] = sign
        hist['above'] = above
        hist['below'] = below
        hist['upper'] = upper
        hist['lower'] = lower
        dic[obs_name] = hist

        np.savetxt(
            os.path.join(directory, 'res', obs_name),
            dat,
            header='Sign: {} {}, above {} {}, below {} {}'.format(
                *sign, *above, *below)
            )

    print("Equal time observables:")
    for obs_name in list_eq:
        print(obs_name)
        if sym_spec is not None:
            symmetry = sym_spec(obs_name, par)
        else:
            symmetry = None
        sign, m_k, e_k, m_k_sum, e_k_sum, m_r, e_r, m_r_sum, e_r_sum, latt = \
            ana_eq(directory, obs_name, sym=symmetry)

        write_res_eq(directory, obs_name,
                     m_k, e_k, m_k_sum, e_k_sum,
                     m_r, e_r, m_r_sum, e_r_sum, latt)

        dic[obs_name+'K'] = m_k
        dic[obs_name+'K_err'] = e_k
        dic[obs_name+'K_sum'] = m_k_sum
        dic[obs_name+'K_sum_err'] = e_k_sum
        dic[obs_name+'R'] = m_r
        dic[obs_name+'R_err'] = e_r
        dic[obs_name+'R_sum'] = m_r_sum
        dic[obs_name+'R_sum_err'] = e_r_sum
        dic[obs_name+'_lattice'] = {
            'L1': latt.L1,
            'L2': latt.L2,
            'a1': latt.a1,
            'a2': latt.a2
            }

    if do_tau:
        print("Time displaced observables:")
        for obs_name in list_tau:
            print(obs_name)
            if sym_spec is not None:
                symmetry = sym_spec(obs_name, par)
            else:
                symmetry = None
            sign, m_k, e_k, m_r0, e_r0, dtau, latt = \
                ana_tau(directory, obs_name, sym=symmetry)

            write_res_tau(directory, obs_name,
                          m_k, e_k, m_r0, e_r0, dtau, latt)

            dic[obs_name+'K'] = m_k
            dic[obs_name+'K_err'] = e_k
            dic[obs_name+'R0'] = m_r0
            dic[obs_name+'R0_err'] = e_r0
            dic[obs_name+'_lattice'] = {
                'L1': latt.L1,
                'L2': latt.L2,
                'a1': latt.a1,
                'a2': latt.a2
                }

    with open(os.path.join(directory, 'res.pkl'), 'wb') as f:
        pickle.dump(dic, f)


def write_res_eq(directory, obs_name,
                 m_k, e_k, m_k_sum, e_k_sum,
                 m_r, e_r, m_r_sum, e_r_sum, latt):
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


def write_res_tau(directory, obs_name, m_k, e_k, m_r0, e_r0, dtau, latt):
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

    np.savetxt(os.path.join(directory, 'res', obs_name, 'R0'),
               np.column_stack([taus, m_r0, e_r0]),
               fmt=['%14.7f', '%16.8f', '%16.8f']
               )
