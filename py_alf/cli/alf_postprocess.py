#!/usr/bin/env python3
"""Analyze Monte Carlo bins."""
# pylint: disable=invalid-name

__author__ = "Jonas Schwab"
__copyright__ = "Copyright 2021-2022, The ALF Project"
__license__ = "GPL"

import os
from argparse import ArgumentParser

from py_alf.utils import find_sim_dirs
from py_alf.check_warmup_tk import check_warmup_tk
from py_alf.check_rebin_tk import check_rebin_tk
from py_alf.analysis import analysis
from py_alf.ana import load_res


def _get_arg_parser():
    parser = ArgumentParser(
        description='Script for postprocessing monte carlo bins.',
        )
    parser.add_argument(
        '--check_warmup', '--warmup', action="store_true",
        help='Check warmup.')
    parser.add_argument(
        '--check_rebin', '--rebin', action="store_true",
        help='Check rebinning for controlling autocorrelation.')
    parser.add_argument(
        '-l', '--check_list', nargs='+', default=None,
        help='List of observables to check for warmup and rebinning.')
    parser.add_argument(
        '--do_analysis', '--ana', action="store_true",
        help='Do analysis.')
    parser.add_argument(
        '--gather', action="store_true",
        help='Gather all analysis results in one file.')
    parser.add_argument(
        '--no_tau', action="store_true",
        help='Skip time displaced correlations.')
    parser.add_argument(
        '--custom_obs', default=os.getenv('ALF_CUSTOM_OBS', None),
        help='File that defines custom observables. '
             '(default: os.getenv("ALF_CUSTOM_OBS", None))')
    parser.add_argument(
        '--symmetry', '--sym', default=None,
        help='File that defines lattice symmetries.')
    parser.add_argument(
        'directories', nargs='*',
        help='Directories to analyze. If empty, analyzes all \
            directories containing file "data.h5" it can find.')
    return parser


if __name__ == '__main__':
    parser = _get_arg_parser()
    args = parser.parse_args()

    if args.custom_obs is None:
        custom_obs = {}
    else:
        with open(os.path.expanduser(args.custom_obs)) as f:
            exec(f.read())

    if args.symmetry is None:
        symmetry = None
    else:
        with open(os.path.expanduser(args.symmetry)) as f:
            exec(f.read())

    if args.directories:
        directories = args.directories
    else:
        directories = find_sim_dirs('.')

    if args.check_warmup and (args.check_list is not None):
        check_warmup_tk(directories, args.check_list, custom_obs=custom_obs)

    if args.check_rebin and (args.check_list is not None):
        check_rebin_tk(directories, args.check_list, custom_obs=custom_obs)

    if args.do_analysis:
        # TODO: Add MPI support
        for directory in directories:
            analysis(directory, custom_obs=custom_obs, symmetry=symmetry,
                     do_tau=not args.no_tau)

    if args.gather:
        df = load_res(directories)
        df.to_pickle('gathered.pkl')
