"""Defines some custom/derived observables."""

import numpy as np

custom_obs = {}

def obs_squared(obs, sign, N_obs):
    """Square of a scalar observable.

    obs.shape = (N_obs,)
    """
    return obs[0]**2 / sign

# Energy squared
custom_obs['E_squared']= {
    'needs': ['Ener_scal'],
    'function': obs_squared,
    'kwargs': {}
}

def E_pot_kin(E_pot_obs, E_pot_sign, E_pot_N_obs,
              E_kin_obs, E_kin_sign, E_kin_N_obs):
    """Ratio of two scalar observables, first observable divided by second."""
    return E_pot_obs/E_kin_obs / (E_pot_sign/E_kin_sign)

# Potential Energy / Kinetic Energy
custom_obs['E_pot_kin']= {
    'needs': ['Pot_scal', 'Kin_scal'],
    'function': E_pot_kin,
    'kwargs': {}
}

def R_k(obs, back, sign, N_orb, N_tau, dtau, latt,
        ks=((0., 0.),), mat=None, NNs=((1, 0), (0, 1), (-1, 0), (0, -1))):
    """Calculate correlation ratio, an RG-invariant quantity derived from
    a correlation function.

    Parameters
    ----------
    obs : array of shape (N_orb, N_orb, N_tau, latt.N)
        Correlation function, the background is already subtracted.
    back : array of shape (N_orb,)
        Background of Correlation function.
    sign : float
        Monte Carlo sign.
    N_orb : int
        Number of orbitals per unit cell.
    N_tau : int
        Number of imaginary time slices. 1 for equal-time correlations.
    dtau : float
        Imaginary time step.
    latt : py_alf.Lattice
        Bravais lattice object.
    ks : list of k-points, default=[(0., 0.)]
        Singular points of the correlation function in the intended order.
    mat : array of shape (N_orb, N_orb), default=None
        Orbital structure of the order parameter. Default: Trace over orbitals.
    NNs : list of tuples, default=[(1, 0), (0, 1), (-1, 0), (0, -1)]
        Deltas in terms of primitive k-vectors of the Bravais lattice.
    """
    if mat is None:
        mat = np.identity(N_orb)
    out = 0
    for k in ks:
        n = latt.k_to_n(k)

        J1 = (obs[..., n].sum(axis=-1) * mat).sum()
        J2 = 0
        for NN in NNs:
            i = latt.nnlistk[n, NN[0], NN[1]]
            J2 += (obs[..., i].sum(axis=-1) * mat).sum() / len(NNs)
        out += (1 - J2/J1)

    return out / len(ks)

# RG-invariant quantity for ferromagnetic order
custom_obs['R_Ferro']= {
    'needs': ['SpinT_eq'],
    'function': R_k,
    'kwargs': {'ks': [(0., 0.)]}
}

# RG-invariant quantity for antiferromagnetic order
custom_obs['R_AFM']= {
    'needs': ['SpinT_eq'],
    'function': R_k,
    'kwargs': {'ks': [(np.pi, np.pi)]}
}

def obs_k(obs, back, sign, N_orb, N_tau, dtau, latt,
          ks=((0., 0.),), mat=None):
    """Mean of correlation function at one, or multiple k-points.

    Calculates integral over tau (=susceptibility) if time-displaced
    correlation is supplied.

    Parameters
    ----------
    obs : array of shape (N_orb, N_orb, N_tau, latt.N)
        Correlation function, the background is already subtracted.
    back : array of shape (N_orb,)
        Background of Correlation function.
    sign : float
        Monte Carlo sign.
    N_orb : int
        Number of orbitals per unit cell.
    N_tau : int
        Number of imaginary time slices. 1 for equal-time correlations.
    dtau : float
        Imaginary time step.
    latt : py_alf.Lattice
        Bravais lattice object.
    ks : list of k-points, default=[(0., 0.)]
    mat : array of shape (N_orb, N_orb), default=None
        Orbital structure. Default: Trace over orbitals.
    """
    if mat is None:
        mat = np.identity(N_orb)
    out = 0
    for k in ks:
        n = latt.k_to_n(k)

        if N_tau == 1:
            out += (obs[:, :, 0, n] * mat).sum()
        else:
            out += (obs[..., n].sum(axis=-1) * mat).sum()*dtau

    return out / len(ks)

# Correlation of Spin z-component at k=(pi, pi)
custom_obs['SpinZ_pipi']= {
    'needs': ['SpinZ_eq'],
    'function': obs_k,
    'kwargs': {'ks': [(np.pi, np.pi)]}
}

# Correlation of Spin x+y-component at k=(pi, pi)
custom_obs['SpinXY_pipi']= {
    'needs': ['SpinXY_eq'],
    'function': obs_k,
    'kwargs': {'ks': [(np.pi, np.pi)]}
}

# Correlation of total Spin at k=(pi, pi)
custom_obs['SpinXYZ_pipi']= {
    'needs': ['SpinT_eq'],
    'function': obs_k,
    'kwargs': {'ks': [(np.pi, np.pi)]}
}
