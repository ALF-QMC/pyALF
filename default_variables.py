"""
Defines dictionaries containing all ALF parameters with default values.

default_params -- Return full set of default parameters for hamiltonian.
params_list -- Return list of parameter names for hamiltonian.
PARAMS_GENERIC -- contains all generic parameters independent from the choice
                  of hamiltonian.
PARAMS_MODEL -- contains namelists dependant on hamiltonian
IN_HAM -- defines which elements of PARAMS_MODEL are needed by a hamiltonian
"""
# pylint: disable = line-too-long

__author__ = "Fakher F. Assaad, and Jonas Schwab"
__copyright__ = "Copyright 2020, The ALF Project"
__license__ = "GPL"

import copy
from collections import OrderedDict


def default_params(ham_name):
    """Return full set of default parameters for hamiltonian."""
    params = OrderedDict()
    for name in IN_HAM[ham_name]:
        params[name] = copy.deepcopy(PARAMS_MODEL[name])
    for name, namelist in PARAMS_GENERIC.items():
        params[name] = copy.deepcopy(namelist)
    return params


def params_list(ham_name, include_generic=False):
    """Return list of parameter names for hamiltonian,
    transformed in all upper case.
    """
    p_list = []
    for name in IN_HAM[ham_name]:
        p_list += list(PARAMS_MODEL[name])
    if include_generic:
        for nlist_name in PARAMS_GENERIC:
            p_list += list(PARAMS_GENERIC[nlist_name])

    return [i.upper() for i in p_list]


PARAMS_GENERIC = OrderedDict()
PARAMS_MODEL = OrderedDict()

IN_HAM = {
    'Hubbard': ["VAR_Lattice", "VAR_Model_Generic", "VAR_Hubbard"],
    'Hubbard_Plain_Vanilla': ["VAR_Lattice", "VAR_Hubbard_Plain_Vanilla"],
    'Kondo': ["VAR_Lattice", "VAR_Model_Generic", "VAR_Kondo"],
    'tV': ["VAR_Lattice", "VAR_Model_Generic", "VAR_tV"],
    'LRC': ["VAR_Lattice", "VAR_Model_Generic", "VAR_LRC"],
    'Z2_Matter': ["VAR_Lattice",  "VAR_Z2_Matter"],
    }

PARAMS_GENERIC["VAR_QMC"] = {
    # General parameters for the Monte Carlo algorithm
    "Nwrap"                : [10,  "Stabilization. Green functions will be computed from scratch after each time interval Nwrap*Dtau."],
    "Nsweep"               : [100, "Number of sweeps per bin."],
    "Nbin"                 : [5, "Number of bins."],
    "Ltau"                 : [1, "1 to calculate time-displaced Green functions; 0 otherwise."],
    "LOBS_ST"              : [0, "Start measurements at time slice LOBS_ST"],
    "LOBS_EN"              : [0, "End measurements at time slice LOBS_EN"],
    "CPU_MAX"              : [0.0, "Code stops after CPU_MAX hours, if 0 or not specified, the code stops after Nbin bins"],
    "Propose_S0"           : [False, "Proposes single spin flip moves with probability exp(-S0)."],
    "Global_moves"         : [False, "Allows for global moves in space and time."],
    "N_global"             : [1, "Number of global moves per sweep."],
    "Global_tau_moves"     : [False, "Allows for global moves on a single time slice."],
    "N_global_tau"         : [1, "Number of global moves that will be carried out on a single time slice."],
    "Nt_sequential_start"  : [0, ""],
    "Nt_sequential_end"    : [-1, ""],
    "Langevin"             : [False, "Langevin update"],
    "Delta_t_Langevin_HMC" : [0.01, "Time step for Langevin or HMC"],
    "Max_Force"            : [1.5,  "Max Force for Langevin" ],
    "HMC"                  : [False, "HMC update"],
    "Leapfrog_steps"       : [0, "Number of leapfrog steps"],
    }

PARAMS_GENERIC["VAR_errors"] = {
    # Post-processing parameters
    "N_skip" : [1, "Number of bins to be skipped."],
    "N_rebin": [1, "Rebinning: Number of bins to combine into one."],
    "N_Cov"  : [0, "If set to 1, covariance computed for time-displaced correlation functions."],
    "N_Back" : [1, "If set to 1, substract background in correlation functions."],
    "N_auto" : [0, "If > 0, calculate autocorrelation."],
    }

PARAMS_GENERIC["VAR_TEMP"] = {
    # Parallel tempering parameters
    "N_exchange_steps"      : [6, "Number of exchange moves."],
    "N_Tempering_frequency" : [10, "The frequency, in units of sweeps, at which the exchange moves are carried out."],
    "mpi_per_parameter_set" : [2, "Number of mpi-processes per parameter set."],
    "Tempering_calc_det"    :
        [True, "Specifies whether the fermion weight has to be taken into \
         account while tempering. Can be set to .F. if the parameters that \
         get varied only enter the Ising action S_0"],
    }

PARAMS_GENERIC["VAR_Max_Stoch"] = {
    # MaxEnt parameters
    "Ngamma"     : [400, "Number of Dirac delta-functions for parametrization."],
    "Om_st"      : [-10.0, "Frequency range lower bound."],
    "Om_en"      : [10.0, "Frequency range upper bound."],
    "Ndis"       : [2000, "Number of boxes for histogram."],
    "NBins"      : [250, "Number of bins for Monte Carlo."],
    "NSweeps"    : [70, "Number of sweeps per bin."],
    "Nwarm"      : [20, "The Nwarm first bins will be ommitted."],
    "N_alpha"    : [14, "Number of temperatures."],
    "alpha_st"   : [1.0, ""],
    "R"          : [1.2, ""],
    "Checkpoint" : [False, ""],
    "Tolerance"  : [0.1, ""],
    }

PARAMS_MODEL["VAR_Lattice"] = {
    # Parameters that define the Bravais lattice
    "L1": [6, ""],
    "L2": [6, ""],
    "Lattice_type": ["Square", ""],
    "Model": ["Hubbard", ""],
    }

PARAMS_MODEL["VAR_Model_Generic"] = {
    # General parameters concerning any model
    "Checkerboard": [True, ""],
    "Symm"        : [True, ""],
    "N_SUN"       : [2, ""],
    "N_FL"        : [1, ""],
    "Phi_X"       : [0.0, ""],
    "Phi_Y"       : [0.0, ""],
    "Bulk"        : [True, ""],
    "N_Phi"       : [0, ""],
    "Dtau"        : [0.1, ""],
    "Beta"        : [5.0, ""],
    "Projector"   : [False, ""],
    "Theta"       : [10.0, ""],
    }

PARAMS_MODEL["VAR_Hubbard"] = {
    # Parameters of the Hubbard hamiltonian
    "Mz"         :  [True, ""] ,
    "ham_T"      :  [1.0, ""]  ,
    "ham_chem"   :  [0.0, ""]  ,
    "ham_U"      :  [4.0, ""]  ,
    "ham_T2"     :  [1.0, ""]  ,
    "ham_U2"     :  [4.0, ""]  ,
    "ham_Tperp"  :  [1.0, ""]  ,
    "Continuous" :  [False, "Continuous HS transformation"],
    }

PARAMS_MODEL["VAR_tV"] = {
    "ham_T"    :  [1.0, ""]  ,
    "ham_chem" :  [0.0, ""]  ,
    "ham_V"    :  [0.5, ""]  ,
    "ham_T2"   :  [1.0, ""]  ,
    "ham_V2"   :  [0.5, ""]  ,
    "ham_Tperp":  [1.0, ""]  ,
    "ham_Vperp":  [0.5, ""]  ,
    }

PARAMS_MODEL["VAR_Hubbard_Plain_Vanilla"] = {
    # Parameters of the Plain Vanilla Hubbard hamiltonian
    "ham_T"       : [1.0, ""]  ,
    "ham_chem"    : [0.0, ""]  ,
    "ham_U"       : [4.0, ""]  ,
    "Dtau"        : [0.1, ""],
    "Beta"        : [5.0, ""],
    "Projector"   : [False, ""],
    "Theta"       : [10.0, ""],
    "Symm"        : [True, ""],
    }


PARAMS_MODEL["VAR_Kondo"] = {
    # Parameters of the SU(N) Kondo lattice
    "ham_T"    :  [1.0, ""]  ,
    "ham_chem" :  [0.0, ""]  ,
    "ham_Uc"   :  [0.0, ""]  ,
    "ham_Uf"   :  [2.0, ""]  ,
    "ham_JK"   :  [2.0, ""]  ,
    }

PARAMS_MODEL["VAR_LRC"] = {
    # Parameters  for the long-ranged Coulomb model
    "ham_T"            :  [1.0, ""]  ,
    "ham_T2"           :  [1.0, ""]  ,
    "ham_Tperp"        :  [1.0, ""]  ,
    "ham_chem"         :  [0.0, ""]  ,
    "ham_U"            :  [4.0, ""]  ,
    "ham_alpha"        :  [0.1, ""]  ,
    "Percent_change"   :  [0.1, ""]  ,
    }

PARAMS_MODEL["VAR_Z2_Matter"] = {
    # Parameters Z2 gauged theories coupled to Z2 matter
    "ham_T"            : [1.0, "Hopping for fermions"],
    "ham_TZ2"          : [1.0, "Hopping for orthogonal fermions"],
    "ham_chem"         : [0.0, "Chemical potential for fermions"],
    "ham_U"            : [0.0, "Hubbard for fermions"],
    "Ham_J"            : [1.0, "Hopping Z2 matter fields"],
    "Ham_K"            : [1.0, "Plaquette term for gauge fields"],
    "Ham_h"            : [1.0, "sigma^x-term for matter"],
    "Ham_g"            : [1.0, "tau^x-term for gauge"],
    "Dtau"             : [0.1,  ""],
    "Beta"             : [10.0, ""],
    "N_SUN"            : [2, ""],
    "Projector"        : [False, ""],
    "Theta"            : [10.0, ""],
    }
