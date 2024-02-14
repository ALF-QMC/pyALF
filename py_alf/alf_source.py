"""
Provides interfaces for compiling, running and postprocessing ALF in Python.
"""
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# py lint: disable=consider-using-f-string

__author__ = "Jonas Schwab"
__copyright__ = "Copyright 2020-2024, The ALF Project"
__license__ = "GPL"

import os
import copy
import subprocess
from collections import OrderedDict
import importlib.util


class cd:
    """Context manager for changing the current working directory."""

    def __init__(self, directory):
        self.directory = os.path.expanduser(directory)
        self.saved_path = os.getcwd()

    def __enter__(self):
        os.chdir(self.directory)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.saved_path)


class ALF_source:
    """
    Objet representing ALF source code.

    Parameters
    ----------
    alf_dir : path-like object, default=os.getenv('ALF_DIR', './ALF')
        Directory containing the ALF source code. If the directory does
        not exist, the source code will be fetched from a server.
        Defaults to environment variable $ALF_DIR if defined, otherwise
        to './ALF'.
    branch : str, optional
        If specified, this will be checked out by git.
    url : str, default='https://git.physik.uni-wuerzburg.de/ALF/ALF.git'
        Address from where to clone ALF if alf_dir does not exist.
    """

    def __init__(self, alf_dir=os.getenv('ALF_DIR', './ALF'), branch=None,
                 url='https://git.physik.uni-wuerzburg.de/ALF/ALF.git'):
        self.alf_dir = os.path.abspath(os.path.expanduser(alf_dir))
        self.branch = branch

        if not os.path.exists(self.alf_dir):
            print("Repository {} does not exist, cloning from {}"
                  .format(alf_dir, url))
            try:
                subprocess.run(["git", "clone", url, self.alf_dir], check=True)
            except subprocess.CalledProcessError as git_clone_failed:
                raise RuntimeError('Error while cloning repository') \
                    from git_clone_failed
        if branch is not None:
            with cd(self.alf_dir):
                print('Checking out branch {}'.format(branch))
                try:
                    subprocess.run(['git', 'checkout', branch], check=True)
                except subprocess.CalledProcessError as git_checkout_failed:
                    raise RuntimeError(
                        'Error while checking out {}'.format(branch)) \
                        from git_checkout_failed

        def import_module(module_name, path):
            """Dynamically import module from given path."""
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        try:
            parse_ham_mod = import_module(
                'parse_ham',
                os.path.join(self.alf_dir, 'Prog', 'parse_ham_mod.py'))
        except FileNotFoundError as parse_ham_not_found:
            raise FileNotFoundError(
                "parse_ham_mod.py not found. "
                "Directory {} ".format(self.alf_dir) +
                "does not contain a supported ALF code.") \
                    from parse_ham_not_found
        try:
            default_parameters_generic = import_module(
                'default_parameters_generic',
                os.path.join(self.alf_dir, 'Prog',
                             'default_parameters_generic.py'))
        except FileNotFoundError as default_parameters_generic_not_found:
            raise FileNotFoundError(
                "default_parameters_generic.py not found. "
                "Directory {} ".format(self.alf_dir) +
                "does not contain a supported ALF code.") \
                    from default_parameters_generic_not_found

        self._PARAMS_GENERIC = default_parameters_generic._PARAMS_GENERIC

        self.default_parameters = get_default_parameters(parse_ham_mod, self.alf_dir)

    def get_ham_names(self):
        """Return list of Hamiltonians."""
        return list(self.default_parameters)

    def get_default_params(self, ham_name, include_generic=True):
        """Return full set of default parameters for hamiltonian."""
        params = OrderedDict()
        for nlist_name, nlist in self.default_parameters[ham_name].items():
            params[nlist_name] = copy.deepcopy(nlist)
        if include_generic:
            for nlist_name, nlist in self._PARAMS_GENERIC.items():
                params[nlist_name] = copy.deepcopy(nlist)
        return params

    def get_params_names(self, ham_name, include_generic=True):
        """Return list of parameter names for hamiltonian,
        transformed in all uppercase.
        """
        p_list = []
        for nlist_name, nlist in self.default_parameters[ham_name].items():
            p_list += list(nlist)
        if include_generic:
            for nlist_name in self._PARAMS_GENERIC:
                p_list += list(self._PARAMS_GENERIC[nlist_name])

        return [i.upper() for i in p_list]

def get_default_parameters(parse_ham_mod, alf_dir):
    """Return dictionary of all default parameters of Hamiltonians.
    By parsing Hamiltonians."""
    try:
        ham_names, ham_files = parse_ham_mod.get_ham_names_ham_files(
            os.path.join(alf_dir, 'Prog', 'Hamiltonians.list')
            )
        ham_files = [os.path.join(alf_dir, 'Prog', ham_file) for
                    ham_file in ham_files]
    except AttributeError:
        # Backwards compatibility fallback
        with open(os.path.join(alf_dir, 'Prog', 'Hamiltonians.list'),
                    'r', encoding='UTF-8') as f:
            ham_names = f.read().splitlines()
        ham_files = [os.path.join(
            alf_dir, 'Prog', 'Hamiltonians',
            'Hamiltonian_{}_smod.F90'.format(ham_name)) for
            ham_name in ham_names]

    default_parameters = {}
    for ham_name, ham_file in zip(ham_names, ham_files):
        # print('Hamiltonian:', ham_name)
        default_parameters[ham_name] = parse_ham_mod.parse(ham_file)
        # pprint.pprint(self.default_parameters[ham_name])
    return default_parameters
