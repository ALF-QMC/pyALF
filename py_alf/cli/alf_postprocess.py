#!/usr/bin/env python3
"""Analyze Monte Carlo bins."""
# pylint: disable=invalid-name

__author__ = "Jonas Schwab"
__copyright__ = "Copyright 2021-2022, The ALF Project"
__license__ = "GPL"

import importlib.util
import os
from argparse import ArgumentParser

from py_alf.ana import load_res
from py_alf.analysis import analysis
from py_alf.check_rebin_tk import check_rebin_tk
from py_alf.check_warmup_tk import check_warmup_tk
from py_alf.utils import find_sim_dirs


def _get_arg_parser():
    parser = ArgumentParser(
        description='Script for postprocessing Monte Carlo bins.',
        )
    parser.add_argument(
        '--check_warmup', '--warmup', action="store_true",
        help='Check warmup. Opens new window.')
    parser.add_argument(
        '--check_rebin', '--rebin', action="store_true",
        help='Check rebinning for controlling autocorrelation. '
             'Opens new window.')
    parser.add_argument(
        '-l', '--check_list', nargs='+', default=None,
        help='List of observables to check for warmup and rebinning.')
    parser.add_argument(
        '--do_analysis', '--ana', action="store_true",
        help='Do analysis.')
    parser.add_argument(
        '--always', action="store_true",
        help='Do not skip analysis if parameters and bins '
             'are older than results.')
    parser.add_argument(
        '--gather', action="store_true",
        help='Gather all analysis results in one file named "gathered.pkl", '
             'representing a pickled pandas DataFrame.')
    parser.add_argument(
        '--no_tau', action="store_true",
        help='Skip time displaced correlations.')
    parser.add_argument(
        '--custom_obs', default=os.getenv('ALF_CUSTOM_OBS', None),
        help='File that defines custom observables. '
             'This file has to define the object custom_obs, '
             'needed by py_alf.analysis. '
             '(default: os.getenv("ALF_CUSTOM_OBS", None))')
    parser.add_argument(
        '--symmetry', '--sym', default=None,
        help='File that defines lattice symmetries. '
             'This file has to define the object symmetry, '
             'needed by py_alf.analysis. (default: None))')
    parser.add_argument(
        'directories', nargs='*',
        help='Directories to analyze. If empty, analyzes all '
            'directories containing file "data.h5" it can find, '
            'starting from the current working directory.')
    return parser


def import_module(module_name, path):
    """Dynamically import module from given path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _main():
    parser = _get_arg_parser()
    args = parser.parse_args()

    if args.custom_obs is None:
        custom_obs = {}
    else:
        try:
            custom_obs_mod = import_module(
                'custom_obs', os.path.expanduser(args.custom_obs))
        except FileNotFoundError as custom_obs_not_found:
            raise FileNotFoundError(f'"{args.custom_obs}" not found.') \
                from custom_obs_not_found
        custom_obs = custom_obs_mod.custom_obs

    if args.symmetry is None:
        symmetry = None
    else:
        try:
            symmetry_mod = import_module(
                'symmetry', os.path.expanduser(args.symmetry))
        except FileNotFoundError as symmetry_not_found:
            raise FileNotFoundError(f'"{args.symmetry}" not found.') \
                from symmetry_not_found
        symmetry = symmetry_mod.symmetry

    directories = args.directories if args.directories else find_sim_dirs('.')

    if args.check_warmup and (args.check_list is not None):
        check_warmup_tk(directories, args.check_list, custom_obs=custom_obs)

    if args.check_rebin and (args.check_list is not None):
        check_rebin_tk(directories, args.check_list, custom_obs=custom_obs)

    if args.do_analysis:
        for directory in directories:
            analysis(directory, custom_obs=custom_obs, symmetry=symmetry,
                     do_tau=not args.no_tau, always=args.always)

    if args.gather:
        df = load_res(directories)
        df.to_pickle('gathered.pkl')


if __name__ == '__main__':
    _main()
