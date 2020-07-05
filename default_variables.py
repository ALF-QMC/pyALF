"""
Defines dictionaries containing all ALF parameters with default values.
PARAMS contains all generic parameters independent from the choice of model.
PARAMS_MODEL contains a dictionary for each model.
"""
# pylint: disable=bad-whitespace

__author__ = "Fakher F. Assaad, and Jonas Schwab"
__copyright__ = "Copyright 2020, The ALF Project"
__license__ = "GPL"

from collections import OrderedDict


PARAMS = OrderedDict()
PARAMS_MODEL = OrderedDict()

PARAMS["VAR_Lattice"] = {
    # Parameters that define the Bravais lattice
    "L1": 6,
    "L2": 6,
    "Lattice_type": "Square",
    "Model": "Hubbard",
    }

PARAMS["VAR_Model_Generic"] = {
    # General parameters concerning any model
    "Checkerboard": True,
    "Symm"        : True,
    "N_SUN"       : 2,
    "N_FL"        : 1,
    "Phi_X"       : 0.0,
    "Phi_Y"       : 0.0,
    "Bulk"        : True,
    "N_Phi"       : 0,
    "Dtau"        : 0.1,
    "Beta"        : 5.0,
    "Projector"   : False,
    "Theta"       : 10.0,
    }

PARAMS["VAR_QMC"] = {
    # General parameters for the Monte Carlo algorithm
    "Nwrap"              : 10 ,
    "NSweep"             : 20 ,
    "NBin"               : 5  ,
    "Ltau"               : 1  ,
    "LOBS_ST"            : 0  ,
    "LOBS_EN"            : 0  ,
    "CPU_MAX"            : 0.0,
    "Propose_S0"         : False,
    "Global_moves"       : False,
    "N_Global"           : 1  ,
    "Global_tau_moves"   : False,
    "N_Global_tau"       : 1  ,
    "Nt_sequential_start": 0  ,
    "Nt_sequential_end"  : -1 ,
    }

PARAMS["VAR_errors"] = {
    # Post-processing parameters
    "n_skip" : 1,
    "N_rebin": 1,
    "N_Cov"  : 0,
    }

PARAMS["VAR_TEMP"] = {
    # Parallel tempering parameters
    "N_exchange_steps"      : 6   ,
    "N_Tempering_frequency" : 10  ,
    "mpi_per_parameter_set" : 2   ,
    "Tempering_calc_det"    : True,
    }

PARAMS["VAR_Max_Stoch"] = {
    # MaxEnt parameters
    "NGamma"     : 400  ,
    "Om_st"      : -10.0,
    "Om_en"      :  10.0,
    "Ndis"       : 2000 ,
    "NBins"      : 250  ,
    "NSweeps"    : 70   ,
    "NWarm"      : 20   ,
    "N_alpha"    : 14   ,
    "alpha_st"   : 1.0  ,
    "R"          : 1.2  ,
    "Channel"    : "P"  ,
    "Checkpoint" : False,
    "Tolerance"  : 0.1  ,
    }

PARAMS_MODEL["VAR_Hubbard"] = {
    # Parameters of the Hubbard hamiltonian
    "HS"       :  "Mz" ,
    "ham_T"    :  1.0  ,
    "ham_chem" :  0.0  ,
    "ham_U"    :  4.0  ,
    "ham_T2"   :  1.0  ,
    "ham_U2"   :  4.0  ,
    "ham_Tperp":  1.0  ,
    }
