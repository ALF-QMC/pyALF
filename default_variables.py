"""
Supplies all ALF parameters with default values.

Default_params -- Object containing all parameters with default values.
"""

__author__ = "Fakher F. Assaad, and Jonas Schwab"
__copyright__ = "Copyright 2020-2022, The ALF Project"
__license__ = "GPL"

import os
import copy
# import pprint
from collections import OrderedDict

from parse_ham import parse


class DefaultParams:
    def __init__(self, alf_dir):
        with open(os.path.join(alf_dir, 'Prog', 'Hamiltonians.list'),
                  'r') as f:
            ham_names = f.read().splitlines()

        self.default_parameters = {}
        for ham_name in ham_names:
            filename = os.path.join(alf_dir, 'Prog', 'Hamiltonians',
                                    'Hamiltonian_{}_smod.F90'.format(ham_name))
            # print('Hamiltonian:', ham_name)

            self.default_parameters[ham_name] = parse(filename)
            # pprint.pprint(self.default_parameters[ham_name])

    def get_ham_names(self):
        """Returns list of Hamiltonians."""
        return list(self.default_parameters)

    def get_params(self, ham_name):
        """Return full set of default parameters for hamiltonian."""
        params = OrderedDict()
        for nlist_name, nlist in self.default_parameters[ham_name].items():
            params[nlist_name] = copy.deepcopy(nlist)
        for nlist_name, nlist in _PARAMS_GENERIC.items():
            params[nlist_name] = copy.deepcopy(nlist)
        return params

    def get_params_names(self, ham_name, include_generic=False):
        """Return list of parameter names for hamiltonian,
        transformed in all upper case.
        """
        p_list = []
        for nlist_name, nlist in self.default_parameters[ham_name].items():
            p_list += list(nlist)
        if include_generic:
            for nlist_name in _PARAMS_GENERIC:
                p_list += list(_PARAMS_GENERIC[nlist_name])

        return [i.upper() for i in p_list]


_PARAMS_GENERIC = OrderedDict([
    ('VAR_QMC',
        {'CPU_MAX': {'comment': 'Code stops after CPU_MAX hours, if 0 or '
                                'not specified, the code stops after '
                                'Nbin bins',
                     'value': 0.0},
         'Delta_t_Langevin_HMC': {'comment': 'Time step for Langevin or '
                                             'HMC',
                                  'value': 0.01},
         'Global_moves': {'comment': 'Allows for global moves in space '
                                     'and time.',
                          'value': False},
         'Global_tau_moves': {'comment': 'Allows for global moves on a '
                                         'single time slice.',
                              'value': False},
         'HMC': {'comment': 'HMC update', 'value': False},
         'LOBS_EN': {'comment': 'End measurements at time slice LOBS_EN',
                     'value': 0},
         'LOBS_ST': {'comment': 'Start measurements at time slice '
                                'LOBS_ST',
                     'value': 0},
         'Langevin': {'comment': 'Langevin update', 'value': False},
         'Leapfrog_steps': {'comment': 'Number of leapfrog steps',
                            'value': 0},
         'Ltau': {'comment': '1 to calculate time-displaced Green '
                             'functions; 0 otherwise.',
                  'value': 1},
         'Max_Force': {'comment': 'Max Force for Langevin', 'value': 1.5},
         'N_global': {'comment': 'Number of global moves per sweep.',
                      'value': 1},
         'N_global_tau': {'comment': 'Number of global moves that will '
                                     'be carried out on a single time '
                                     'slice.',
                          'value': 1},
         'Nbin': {'comment': 'Number of bins.', 'value': 5},
         'Nsweep': {'comment': 'Number of sweeps per bin.', 'value': 100},
         'Nt_sequential_end': {'comment': '', 'value': -1},
         'Nt_sequential_start': {'comment': '', 'value': 0},
         'Nwrap': {'comment': 'Stabilization. Green functions will be '
                              'computed from scratch after each time '
                              'interval Nwrap*Dtau.',
                   'value': 10},
         'Propose_S0': {'comment': 'Proposes single spin flip moves with '
                                   'probability exp(-S0).',
                        'value': False}}),
    ('VAR_errors',
        {'N_Back': {'comment': 'If set to 1, substract background in '
                               'correlation functions.',
                    'value': 1},
         'N_Cov': {'comment': 'If set to 1, covariance computed for '
                              'time-displaced correlation functions.',
                   'value': 0},
         'N_auto': {'comment': 'If > 0, calculate autocorrelation.',
                    'value': 0},
         'N_rebin': {'comment': 'Rebinning: Number of bins to combine '
                                'into one.',
                     'value': 1},
         'N_skip': {'comment': 'Number of bins to be skipped.',
                    'value': 1}}),
    ('VAR_TEMP',
        {'N_Tempering_frequency': {'comment': 'The frequency, in units '
                                              'of sweeps, at which the '
                                              'exchange moves are '
                                              'carried out.',
                                   'value': 10},
         'N_exchange_steps': {'comment': 'Number of exchange moves.',
                              'value': 6},
         'Tempering_calc_det': {'comment': 'Specifies whether the '
                                           'fermion weight has to be '
                                           'taken into          account '
                                           'while tempering. Can be set '
                                           'to .F. if the parameters '
                                           'that          get varied '
                                           'only enter the Ising action '
                                           'S_0',
                                'value': True},
         'mpi_per_parameter_set': {'comment': 'Number of mpi-processes '
                                              'per parameter set.',
                                   'value': 2}}),
    ('VAR_Max_Stoch',
        {'Checkpoint': {'comment': '', 'value': False},
         'NBins': {'comment': 'Number of bins for Monte Carlo.',
                   'value': 250},
         'NSweeps': {'comment': 'Number of sweeps per bin.', 'value': 70},
         'N_alpha': {'comment': 'Number of temperatures.', 'value': 14},
         'Ndis': {'comment': 'Number of boxes for histogram.',
                  'value': 2000},
         'Ngamma': {'comment': 'Number of Dirac delta-functions for '
                               'parametrization.',
                    'value': 400},
         'Nwarm': {'comment': 'The Nwarm first bins will be ommitted.',
                   'value': 20},
         'Om_en': {'comment': 'Frequency range upper bound.',
                   'value': 10.0},
         'Om_st': {'comment': 'Frequency range lower bound.',
                   'value': -10.0},
         'R': {'comment': '', 'value': 1.2},
         'Tolerance': {'comment': '', 'value': 0.1},
         'alpha_st': {'comment': '', 'value': 1.0}})
    ])
