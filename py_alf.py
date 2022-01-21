"""Provides interfaces for compilig, running and postprocessing ALF in Python.
"""
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=consider-using-f-string

__author__ = "Jonas Schwab"
__copyright__ = "Copyright 2020-2022, The ALF Project"
__license__ = "GPL"

import os
import re
import copy
import subprocess
import shutil
from collections import OrderedDict
import importlib.util

import numpy as np
import pandas as pd

from default_parameters_generic import _PARAMS_GENERIC
from alf_ana.check_warmup import check_warmup
from alf_ana.check_rebin import check_rebin
from alf_ana.analysis import analysis
from alf_ana.ana import load_res


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

    Optional arguments:
    alf_dir -- Directory containing the ALF source code. If the directory does
               not exist, the source cFalseode will be fetched from a server.
               Defaults to environment variable $ALF_DIR if present, otherwise
               to './ALF'.
    branch  -- If specified, this will be checked out by git.
    url     -- Address, from where to clone ALF if alf_dir not exists
               (default: 'https://git.physik.uni-wuerzburg.de/ALF/ALF.git')
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
                raise Exception('Error while cloning repository') \
                    from git_clone_failed
        if branch is not None:
            with cd(self.alf_dir):
                print('Checking out branch {}'.format(branch))
                try:
                    subprocess.run(['git', 'checkout', branch], check=True)
                except subprocess.CalledProcessError as git_checkout_failed:
                    raise Exception(
                        'Error while checking out {}'.format(branch)) \
                        from git_checkout_failed

        def import_module(module_name, path):
            """Dynamically import module from given path."""
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        
        parse_ham = import_module(
            'parse_ham', os.path.join(self.alf_dir, 'Prog', 'parse_ham.py'))

        # Parse ALF Hamiltonians to get parameter list.
        with open(os.path.join(self.alf_dir, 'Prog', 'Hamiltonians.list'),
                  'r', encoding='UTF-8') as f:
            ham_names = f.read().splitlines()

        self.default_parameters = {}
        for ham_name in ham_names:
            filename = os.path.join(self.alf_dir, 'Prog', 'Hamiltonians',
                                    'Hamiltonian_{}_smod.F90'.format(ham_name))
            # print('Hamiltonian:', ham_name)
            self.default_parameters[ham_name] = parse_ham.parse(filename)
            # pprint.pprint(self.default_parameters[ham_name])

    def get_ham_names(self):
        """Returns list of Hamiltonians."""
        return list(self.default_parameters)

    def get_default_params(self, ham_name):
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


class Simulation:
    """Object corresponding to a simulation directory.

    Provides functions for preparing, running, and postprocessing a simulation.

    Required arguments:
    alf_src  -- Instance of ALF_source
    ham_name -- Name of the hamiltonian
    sim_dict -- Dictionary specfying parameters owerwriting defaults.
                Can be a list of dictionaries to enable parallel tempering.

    Optional arguments:
    sim_dir -- Directory in which the Monte Carlo will be run.
               If not specified, sim_dir will be generated from sim_dict.
    sim_root-- Directory to prepend to sim_dir. (default: "ALF_data")
    mpi     -- Employ MPI (default: False)
    n_mpi   -- Number of MPI processes
    n_omp   -- Number of OpenMP threads per process (default: 1)
    mpiexec -- Command used for starting a MPI run (default: "mpiexec")
    machine -- Possible values: GNU, INTEL, PGI, SUPERMUC-NG, JUWELS
               default: GNU
    stab    -- Which version of stabilisation to employ
               Possible values: STAB1, STAB2, STAB3, LOG
    devel   -- Compile with additional flags for development and debugging
               default: False
    hdf5    -- Whether to compile ALF with HDF5 (default: True)
               Full postprocessing support only exists with HDF5.
    machine and stab are not case sensitive.
    """

    def __init__(self, alf_src, ham_name, sim_dict, **kwargs):
        if not isinstance(alf_src, ALF_source):
            raise Exception('alf_src needs to be an instance of ALF_source')
        self.alf_src = alf_src
        self.ham_name = ham_name
        self.sim_dict = sim_dict
        self.sim_dir = os.path.abspath(os.path.expanduser(os.path.join(
            kwargs.pop("sim_root", "ALF_data"),
            kwargs.pop("sim_dir",
                       directory_name(alf_src, ham_name, sim_dict)))))
        self.mpi = kwargs.pop("mpi", False)
        self.n_mpi = kwargs.pop("n_mpi", None)
        self.n_omp = kwargs.pop('n_omp', 1)
        self.mpiexec = kwargs.pop('mpiexec', 'mpiexec')
        stab = kwargs.pop('stab', '').upper()
        machine = kwargs.pop('machine', 'GNU').upper()
        self.devel = kwargs.pop('devel', False)
        self.hdf5 = kwargs.pop('hdf5', True)
        if kwargs:
            raise Exception('Unused keyword arguments: {}'.format(kwargs))

        self.tempering = isinstance(sim_dict, list)
        if self.tempering:
            self.mpi = True

        # Check if all parameters in sim_dict are defined in default_variables
        p_list = self.alf_src.get_params_names(
            self.ham_name, include_generic=True)

        if self.tempering:
            for sim_dict0 in self.sim_dict:
                for par_name in sim_dict0:
                    if par_name.upper() not in p_list:
                        raise Exception(
                            'Parameter {} not listet in default_variables'
                            .format(par_name))
        else:
            for par_name in self.sim_dict:
                if par_name.upper() not in p_list:
                    raise Exception(
                        'Parameter {} not listet in default_variables'
                        .format(par_name))

        if self.mpi and self.n_mpi is None:
            raise Exception('You have to specify n_mpi if you use MPI.')

        if machine not in ['GNU', 'INTEL', 'PGI', 'JUWELS', 'SUPERMUC-NG']:
            raise Exception('Illegal value machine={}'.format(machine))

        if stab not in ['STAB1', 'STAB2', 'STAB3', 'LOG', '']:
            raise Exception('Illegal value stab={}'.format(stab))

        self.config = '{} {}'.format(machine, stab).strip()

        if self.mpi:
            self.config += ' MPI'
        else:
            self.config += ' NOMPI'

        if self.tempering:
            self.config += ' TEMPERING'

        if self.devel:
            self.config += ' DEVEL'

        if self.hdf5:
            self.config += ' HDF5'

        self.config += ' NO-INTERACTIVE'

        self.custom_obs = {}

    def compile(self):
        """Compile ALF. Clones a new repository if alf_dir does not exist."""
        compile_alf(self.alf_src.alf_dir, config=self.config)

    def run(self, copy_bin=False, only_prep=False):
        """Prepars simulation directory and run ALF.

        Optional arguments:
        copy_bin  -- Copy ALF binary into simulation folder (default: False)
        only_prep -- Do not run ALF, but only prepare directory (default: False)
        """
        if self.tempering:
            _prep_sim_dir(self.alf_src, self.sim_dir,
                          self.ham_name, self.sim_dict[0])
            for i, sim_dict in enumerate(self.sim_dict):
                _prep_sim_dir(self.alf_src,
                              os.path.join(self.sim_dir, "Temp_{}".format(i)),
                              self.ham_name, sim_dict)
        else:
            _prep_sim_dir(self.alf_src, self.sim_dir,
                          self.ham_name, self.sim_dict)

        env = getenv(self.config, self.alf_src.alf_dir)
        env['OMP_NUM_THREADS'] = str(self.n_omp)
        executable = os.path.join(self.alf_src.alf_dir, 'Prog', 'ALF.out')
        if copy_bin:
            shutil.copy(executable, self.sim_dir)
            executable = os.path.join(self.sim_dir, 'ALF.out')
        if only_prep:
            return
        with cd(self.sim_dir):
            print('Run {}'.format(executable))
            try:
                if self.mpi:
                    command = [self.mpiexec, '-n', str(self.n_mpi), executable]
                else:
                    command = executable
                subprocess.run(command, check=True, env=env)
            except subprocess.CalledProcessError as ALF_crash:
                print('Error while running {}.'.format(executable))
                print('parameters:')
                with open('parameters', 'r') as f:
                    print(f.read())
                raise Exception('Error while running {}.'.format(executable)) \
                    from ALF_crash

    def get_directories(self):
        """Return list of directories connected to this simulation."""
        if self.tempering:
            directories = [os.path.join(self.sim_dir, "Temp_{}".format(i))
                           for i in range(len(self.sim_dict))]
        else:
            directories = [self.sim_dir]
        return directories

    def check_warmup(self, names):
        """Plot bins to determine n_skip.
        names: Names of Observables to check
        """
        check_warmup(self.get_directories(), names, custom_obs=self.custom_obs)

    def check_rebin(self, names):
        """Plot error vs n_rebin to control autocorrelation.
        names: Names of Observables to check
        """
        check_rebin(self.get_directories(), names, custom_obs=self.custom_obs)

    def analysis(self, python_version=True, symmetry=None):
        """Performs default analysis on Monte Carlo data.

        The non-python version is legacy and does not support all
        postprocessing features.
        """
        for directory in self.get_directories():
            if python_version:
                analysis(directory,
                         custom_obs=self.custom_obs, symmetry=symmetry)
            else:
                analysis_fortran(self.alf_src.alf_dir, directory,
                                 hdf5=self.hdf5)

    def get_obs(self, python_version=True):
        """Returns dictionary containing anaysis results from observables.

        The non-python version is legacy and does not support all
        postprocessing features, e.g. time-displaced observables.
        """
        if python_version:
            return load_res(self.get_directories())

        dicts = {}
        for directory in self.get_directories():
            dicts[directory] = get_obs(directory, names=None)
        return pd.DataFrame(dicts).transpose()

    def print_info_file(self):
        """Print info file(s) that get generated by ALF."""
        for directory in self.get_directories():
            filename = os.path.join(directory, 'info')
            if os.path.exists(filename):
                print('===== {} ====='.format(filename))
                with open(filename, 'r') as f:
                    print(f.read())
            else:
                print('{} does not exist.'.format(filename))
                return


def _prep_sim_dir(alf_src, sim_dir, ham_name, sim_dict):
    print('Prepare directory "{}" for Monte Carlo run.'.format(sim_dir))
    if not os.path.exists(sim_dir):
        print('Create new directory.')
        os.makedirs(sim_dir)

    with cd(sim_dir):
        if 'confout_0' in os.listdir() or 'confout_0.h5' in os.listdir():
            print('Resuming previous run.')
        shutil.copyfile(os.path.join(
            alf_src.alf_dir, 'Scripts_and_Parameters_files', 'Start', 'seeds'),
                 'seeds')
        params = set_param(alf_src, ham_name, sim_dict)
        write_parameters(params)
        out_to_in(verbose=False)


def _convert_par_to_str(parameter):
    """Converts a given parameter value to a string that can be
    written into a parameter file.
    """
    if isinstance(parameter, bool):
        if parameter:
            return '.T.'
        return '.F.'
    if isinstance(parameter, float):
        if 'e' in '{}'.format(parameter):
            return '{}'.format(parameter).replace('e', 'd')
        return '{}d0'.format(parameter)
    if isinstance(parameter, int):
        return '{}'.format(parameter)
    if isinstance(parameter, str):
        return '"{}"'.format(parameter)

    raise Exception('Error in "_convert_par_to_str": unrecognized type')


def write_parameters(params):
    """Writes nameslists to file 'parameters'"""
    with open('parameters', 'w') as file:
        for namespace in params:
            file.write("&{}\n".format(namespace))
            for var in params[namespace]:
                file.write('{} = {}  ! {}\n'.format(
                    var,
                    _convert_par_to_str(params[namespace][var]['value']),
                    params[namespace][var]['comment']
                    ))
            file.write("/\n\n")


def directory_name(alf_src, ham_name, sim_dict):
    """Returns name of directory for simulations, given a set of simulation
    parameters.
    """
    p_list = alf_src.get_params_names(ham_name)
    if isinstance(sim_dict, list):
        sim_dict = sim_dict[0]
        dirname = 'temper_{}_'.format(ham_name)
    else:
        dirname = '{}_'.format(ham_name)
    for name, value in sim_dict.items():
        if name.upper() in p_list:
            if name.upper() == 'MODEL':
                if value != ham_name:
                    dirname = '{}{}_'.format(dirname, value)
            elif name.upper() == "LATTICE_TYPE":
                dirname = '{}{}_'.format(dirname, value)
            else:
                if name.upper().startswith('HAM_'):
                    name_temp = name[4:]
                else:
                    name_temp = name
                dirname = '{}{}={}_'.format(dirname, name_temp, value)
    return dirname[:-1]


def _update_var(params, var, value):
    """Try to update value of parameter called var in params."""
    for name in params:
        for var2 in params[name]:
            if var2.lower() == var.lower():
                params[name][var2]['value'] = value
                return params
    raise Exception('"{}" does not correspond to a parameter'.format(var))


def set_param(alf_src, ham_name, sim_dict):
    """Returns dictionary containing all parameters needed by ALF.

    Input: Dictionary with chosen set of <parameter: value> pairs.
    Output: Dictionary containing all namelists needed by ALF.
    """
    params = alf_src.get_default_params(ham_name)

    params["VAR_ham_name"] = {
        "ham_name": {'value': ham_name, 'comment': "Name of Hamiltonian"}
    }

    for name, value in sim_dict.items():
        params = _update_var(params, name, value)
    return params


def getenv(config, alf_dir='.'):
    """Get environment variables for compiling ALF."""
    with cd(alf_dir):
        subprocess.run(
            ['bash', '-c',
             '. ./configure.sh {} > /dev/null || exit 1 && env > environment'
             .format(config)],
            check=True)
        with open('environment', 'r') as f:
            lines = f.readlines()
    env = {}
    for line in lines:
        if (not re.search(r"^BASH_FUNC.*%%=()", line)) and '=' in line:
            item = line.strip().split("=", 1)
            if len(item) == 2:
                env[item[0]] = item[1]
            else:
                env[item[0]] = ''
    return env


def compile_alf(alf_dir='ALF', branch=None, config='GNU noMPI',
                url='https://git.physik.uni-wuerzburg.de/ALF/ALF.git'):
    """Compile ALF. Clone a new repository if alf_dir does not exist."""

    alf_dir = os.path.abspath(alf_dir)
    if not os.path.exists(alf_dir):
        print("Repository {} does not exist, cloning from {}"
              .format(alf_dir, url))
        try:
            subprocess.run(["git", "clone", url, alf_dir], check=True)
        except subprocess.CalledProcessError as git_clone_failed:
            raise Exception('Error while cloning repository') \
                from git_clone_failed

    with cd(alf_dir):
        if branch is not None:
            print('Checking out branch {}'.format(branch))
            try:
                subprocess.run(['git', 'checkout', branch], check=True)
            except subprocess.CalledProcessError as git_checkout_failed:
                raise Exception('Error while checking out {}'.format(branch)) \
                    from git_checkout_failed
        env = getenv(config)
        print('Compiling ALF... ', end='')
        subprocess.run(['make', 'clean'], check=True, env=env)
        subprocess.run(['make', 'ana'], check=True, env=env)
        subprocess.run(['make', 'program'], check=True, env=env)
        print('Done.')


def out_to_in(verbose=False):
    """Renames all the output configurations confout_* to confin_*
    to continue the Monte Carlo simulation where the previous stopped.
    """
    for name in os.listdir():
        if name.startswith('confout_'):
            name2 = 'confin_' + name[8:]
            if verbose:
                print('mv {} {}'.format(name, name2))
            os.replace(name, name2)


def analysis_fortran(alf_dir, sim_dir='.', hdf5=False):
    """Perform the default analysis unsing ALFs own analysis routines
    on all files ending in _scal, _eq or _tau in directory sim_dir. Not fully
    supported
    """
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    with cd(sim_dir):
        if hdf5:
            executable = os.path.join(alf_dir, 'Analysis', 'ana_hdf5.out')
            subprocess.run([executable], check=True, env=env)
        else:
            for name in os.listdir():
                if name.endswith('_scal'):
                    print('Analysing {}'.format(name))
                    executable = os.path.join(alf_dir, 'Analysis', 'ana.out')
                    subprocess.run([executable, name], check=True, env=env)

            for name in os.listdir():
                if name.endswith('_eq'):
                    print('Analysing {}'.format(name))
                    executable = os.path.join(alf_dir, 'Analysis', 'ana.out')
                    subprocess.run([executable, name], check=True, env=env)

            for name in os.listdir():
                if name.endswith('_tau'):
                    print('Analysing {}'.format(name))
                    executable = os.path.join(alf_dir, 'Analysis', 'ana.out')
                    subprocess.run([executable, name], check=True, env=env)


def get_obs(sim_dir, names=None):
    """Returns dictionary containing analysis results from observables.

    Currently only scalar and equal time correlators.
    If names is None: gets all observables, else the ones listed in names
    """
    obs = {}
    if names is None:
        names = os.listdir(sim_dir)
    for name in names:
        if name.endswith('_scalJ'):
            name0 = name[:-1]
            temp = _read_scalJ(os.path.join(sim_dir, name))
            obs[name0+'_sign'] = temp['sign'][0]
            obs[name0+'_sign_err'] = temp['sign'][1]
            for i, temp2 in enumerate(temp['obs']):
                name2 = '{}{}'.format(name0, i)
                obs[name2] = temp['obs'][i, 0]
                obs[name2+'_err'] = temp['obs'][i, 1]
        if name.endswith('_eqJK'):
            name0 = name[:-2]+name[-1]
            temp = _read_eqJ(os.path.join(sim_dir, name))
            obs[name0] = temp['dat'][..., 0] + 1j*temp['dat'][..., 1]
            obs[name0+'_err'] = temp['dat'][..., 2] + 1j*temp['dat'][..., 3]
            obs[name0+'_k'] = temp['k']
        if name.endswith('_eqJR'):
            name0 = name[:-2]+name[-1]
            temp = _read_eqJ(os.path.join(sim_dir, name))
            obs[name0] = temp['dat'][..., 0] + 1j*temp['dat'][..., 1]
            obs[name0+'_err'] = temp['dat'][..., 2] + 1j*temp['dat'][..., 3]
            obs[name0+'_r'] = temp['r']
    return obs


def _read_scalJ(name):
    """Returns dictionary containing anaysis results from scalar observable.
    """
    with open(name) as f:
        lines = f.readlines()
    N_obs = int((len(lines)-2)/2)

    sign = np.loadtxt(lines[-1].split()[-2:])
    print(name, N_obs)

    obs = np.zeros([N_obs, 2])
    for iobs in range(N_obs):
        obs[iobs] = lines[2*iobs+2].split()[-2:]

    return {'sign': sign, 'obs': obs}


def _read_eqJ(name):
    """Returns dictionary containing analysis results from equal time
    correlation function
    """
    with open(name) as f:
        lines = f.readlines()

    if name.endswith('K'):
        x_name = 'k'
    elif name.endswith('R'):
        x_name = 'r'

    N_lines = len(lines)
    N_orb = None
    for i in range(1, N_lines):
        if len(lines[i].split()) == 2:
            N_orb = int(np.sqrt(i-1))
            break
    if N_orb is None:
        N_orb = int(np.sqrt(i-1))

    N_x = int(N_lines / (1 + N_orb**2))

    dat = np.empty([N_x, N_orb, N_orb, 4])
    x = np.empty([N_x, 2])

    for i_x in range(N_x):
        x[i_x] = np.loadtxt(lines[i_x*(1 + N_orb**2)].split())
        for i_orb1 in range(N_orb):
            for i_orb2 in range(N_orb):
                dat[i_x, i_orb1, i_orb2] = np.loadtxt(
                    lines[i_x*(1+N_orb**2)+1+i_orb1*N_orb+i_orb2].split()[2:])

    return {x_name: x, 'dat': dat}
