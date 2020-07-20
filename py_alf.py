"""Provides interfaces for compilig, running and postprocessing ALF in Python.
"""
# pylint: disable=invalid-name, too-many-arguments, too-many-branches
# pylint: disable=too-many-instance-attributes

__author__ = "Jonas Schwab"
__copyright__ = "Copyright 2020, The ALF Project"
__license__ = "GPL"

import os
import subprocess
from shutil import copyfile
import numpy as np
import default_variables


class cd:
    """Context manager for changing the current working directory."""

    def __init__(self, directory):
        self.directory = os.path.expanduser(directory)
        self.saved_path = os.getcwd()

    def __enter__(self):
        os.chdir(self.directory)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.saved_path)


class Simulation:
    """Object corresponding to a simulation directory.

    Provides functions for preparing, running, and postprocessing a simulation.
    """

    def __init__(self, sim_dict, alf_dir='./ALF', sim_dir=None,
                 executable=None, compile_config='GNU noMPI', branch=None,
                 mpi=False, n_mpi=None):
        """Initialize the Simulation object.

        Required argument:
        sim_dict -- Dictionary specfying parameters owerwriting defaults.
                    Can be a list of dictionaries to enable parallel tempering.

        Keyword arguments:
        alf_dir -- Directory containing the ALF source code (default: './ALF').
                   If the directory does not exist, the source code will be
                   fetched from a server.
        sim_dir -- Directory in which the Monte Carlo will be run.
                   If not specified, sim_dir will be generated from sim_dict.
        executable -- Name of ALF executable to run.
                      If not specified, executable will be the model name.
        compile_config
            -- Arguments to hand over to configure script prior to compilation.
               (default: 'GNU noMPI')
        branch -- If specified, this will be checked out, prior to compilation.
        mpi    -- Employ MPI (default: False)
        n_mpi  -- Number of MPI processes
        """
        self.mpi = mpi
        if isinstance(sim_dict, list):
            self.tempering = True
            self.mpi = True
        else:
            self.tempering = False

        if self.mpi and n_mpi is None:
            raise Exception('You have to specify n_mpi if you use MPI.')
        self.n_mpi = n_mpi

        self.sim_dict = sim_dict
        self.alf_dir = os.path.abspath(os.path.expanduser(alf_dir))
        if sim_dir is None:
            self.sim_dir = os.path.abspath(directory_name(sim_dict))
        else:
            self.sim_dir = os.path.abspath(sim_dir)

        if executable is None:
            executable = sim_dict['Model']
        self.executable = executable
        self.compile_config = compile_config.upper()
        if self.mpi:
            self.compile_config = self.compile_config.replace('NOMPI', 'MPI')
            self.compile_config = self.compile_config.replace('SERIAL', 'MPI')
            if 'MPI' not in self.compile_config:
                self.compile_config = self.compile_config + ' MPI'
        if self.tempering and 'TEMPERING' not in self.compile_config:
            self.compile_config = self.compile_config + ' TEMPERING'
        self.branch = branch

    def compile(self):
        """Compiles ALF. Clones a new repository if alf_dir does not exist."""
        compile_alf(self.alf_dir, self.branch, self.compile_config,
                    model='all')

    def run(self):
        """Prepares simulation directory and runs ALF."""
        if self.tempering:
            _prep_sim_dir(self.alf_dir, self.sim_dir, self.sim_dict[0])
            for i, sim_dict in enumerate(self.sim_dict):
                _prep_sim_dir(self.alf_dir,
                              os.path.join(self.sim_dir, "Temp_{}".format(i)),
                              sim_dict)
        else:
            _prep_sim_dir(self.alf_dir, self.sim_dir, self.sim_dict)

        executable = os.path.join(self.alf_dir, 'Prog', self.executable+'.out')
        with cd(self.sim_dir):
            print('Run {}'.format(executable))
            try:
                if self.mpi:
                    subprocess.run(
                        ['mpiexec', '-n', str(self.n_mpi), executable],
                        check=True)
                else:
                    subprocess.run(executable, check=True)
            except subprocess.CalledProcessError:
                print('Error while running {}.'.format(executable))
                with open('parameters') as f:
                    print(f.read())

    def analysis(self):
        """Performs default analysis on Monte Carlo data."""
        if self.tempering:
            for i in range(len(self.sim_dict)):
                analysis(self.alf_dir,
                         os.path.join(self.sim_dir, "Temp_{}".format(i)))
        else:
            analysis(self.alf_dir, self.sim_dir)

    def get_obs(self, names=None):
        """Returns dictionary containing anaysis results from observables.

        Currently only scalar and equal time correlators.
        If names is None: gets all observables, else the ones listed in names.
        """
        return get_obs(self.sim_dir, names)


def _prep_sim_dir(alf_dir, sim_dir, sim_dict):
    print('Prepare directory "{}" for Monte Carlo run.'.format(sim_dir))
    if not os.path.exists(sim_dir):
        os.mkdir(sim_dir)

    with cd(sim_dir):
        copyfile(os.path.join(alf_dir, 'Scripts_and_Parameters_files', 'Start',
                              'seeds'),
                 'seeds')
        params = set_param(sim_dict)
        write_parameters(params)
        out_to_in(verbose=False)


def _convert_par_to_str(parameter):
    """Converts a given parameter value to a string that can be
    written into a parameter file.
    """
    if isinstance(parameter, (float, int)):
        return str(parameter)
    if isinstance(parameter, str):
        return '"' + parameter + '"'
    if isinstance(parameter, bool):
        if parameter:
            return '.T.'
        return '.F.'

    raise Exception('Error in "_convert_par_to_str": unrecognized type')


def write_parameters(params):
    """Writes nameslists to file 'parameters'"""
    with open('parameters', 'w') as file:
        for namespace in params:
            file.write("&{}\n".format(namespace))
            for var in params[namespace]:
                file.write(var + ' = '
                           + _convert_par_to_str(params[namespace][var])
                           + '\n')
            file.write("/\n\n")


def directory_name(sim_dict):
    """Returns name of directory for simulations, given a set of simulation
    parameters.

    TODO: Automatically generate a list of all parameters to use.
    """
    dirname = ''
    if isinstance(sim_dict, list):
        sim_dict = sim_dict[0]
        dirname = 'temper_'
    for name in sim_dict:
        if name in ["L1", "L2", "Lattice_type", "Model",
                    "Checkerboard", "Symm", "N_SUN", "N_FL", "Phi_X", "N_Phi",
                    "Dtau", "Beta", "Projector",
                    "Theta", "ham_T", "ham_chem", "ham_U",
                    "ham_T2", "ham_U2", "ham_Tperp"]:
            if name in ["Lattice_type", "Model"]:
                dirname = '{}{}_'.format(dirname, sim_dict[name])
            else:
                dirname = '{}{}={}_'.format(dirname, name.strip("ham_"),
                                            sim_dict[name])
    return dirname[:-1]


def _update_var(params, var, value):
    """Try to update value of parameter called var in params."""
    for name in params:
        for var2 in params[name]:
            if var2.lower() == var.lower():
                params[name][var2] = value
                return params
    raise Exception('"{}" does not correspond to a parameter'.format(var))


def set_param(sim_dict):
    """Returns dictionary containing all parameters needed by ALF.

    Input: Dictionary with chosen set of <parameter: value> pairs.
    Output: Dictionary containing all namelists needed by ALF.
    """
    model = sim_dict['Model']
    params = default_variables.PARAMS
    params_model = default_variables.PARAMS_MODEL
    params['VAR_'+model] = params_model['VAR_'+model]

    for name, value in sim_dict.items():
        params = _update_var(params, name, value)
    return params


def compile_alf(alf_dir='ALF', branch=None, config='GNU noMPI', model='all',
                url='git@git.physik.uni-wuerzburg.de:ALF/ALF.git'):
    """Compiles ALF. Clones a new repository if alf_dir does not exist."""

    alf_dir = os.path.abspath(alf_dir)
    if not os.path.exists(alf_dir):
        print("Repository {} does not exist, cloning from {}"
              .format(alf_dir, url))
        try:
            subprocess.run(["git", "clone", url, alf_dir], check=True)
        except subprocess.CalledProcessError:
            print('Error while cloning repository')

    with cd(alf_dir):
        if branch is not None:
            try:
                subprocess.run(["git", "checkout", branch], check=True)
            except subprocess.CalledProcessError:
                print('Error while checking out {}'.format(branch))
        subprocess.run(
            ['bash', '-c',
             '. ./configureHPC.sh {}; env > environment'.format(config)],
            check=True)
        with open('environment', 'r') as f:
            lines = f.readlines()
        print(lines)
        env = dict((line.strip().split("=", 1) for line in lines))
        subprocess.run(['make', 'clean'], check=True, env=env)
        subprocess.run(['make', model], check=True, env=env)


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


def analysis(alf_dir, sim_dir='.'):
    """Perform the default analysis on all files ending in _scal, _eq or _tau
    in directory sim_dir.
    """
    with cd(sim_dir):
        if os.path.exists('Var_scal'):
            os.remove('Var_scal')
        for name in os.listdir():
            if name.endswith('_scal'):
                print('Analysing {}'.format(name))
                os.symlink(name, 'Var_scal')
                executable = os.path.join(alf_dir, 'Analysis', 'cov_scal.out')
                subprocess.run(executable, check=True)
                os.remove('Var_scal')
                os.replace('Var_scalJ', name+'J')

                for name2 in os.listdir():
                    if name2.startswith('Var_scal_Auto_'):
                        name3 = name + name2[8:]
                        os.replace(name2, name3)

        if os.path.exists('ineq'):
            os.remove('ineq')
        for name in os.listdir():
            if name.endswith('_eq'):
                print('Analysing {}'.format(name))
                os.symlink(name, 'ineq')
                executable = os.path.join(alf_dir, 'Analysis', 'cov_eq.out')
                subprocess.run(executable, check=True)
                os.remove('ineq')
                os.replace('equalJ', name+'JK')
                os.replace('equalJR', name+'JR')

                for name2 in os.listdir():
                    if name2.startswith('Var_eq_Auto_Tr'):
                        name3 = name + name2[6:]
                        os.replace(name2, name3)

        if os.path.exists('intau'):
            os.remove('intau')
        for name in os.listdir():
            if name.endswith('_tau'):
                print('Analysing {}'.format(name))
                os.symlink(name, 'intau')
                executable = os.path.join(alf_dir, 'Analysis', 'cov_tau.out')
                subprocess.run(executable, check=True)
                os.remove('intau')
                os.replace('SuscepJ', name+'JK')

                for name2 in os.listdir():
                    if name2.startswith('g_'):
                        directory = name[:-4] + name2[1:]
                        if not os.path.exists(directory):
                            os.mkdir(directory)
                        os.replace(name2, os.path.join(directory, name2))


def get_obs(sim_dir, names=None):
    """Returns dictionary containing anaysis results from observables.

    Currently only scalar and equal time correlators.
    If names is None: gets all observables, else the ones listed in names
    """
    obs = {}
    if names is None:
        names = os.listdir(sim_dir)
    for name in names:
        if name.endswith('_scalJ'):
            obs[name] = _read_scalJ(os.path.join(sim_dir, name))
        if name.endswith('_eqJK') or name.endswith('_eqJR'):
            obs[name] = _read_eqJ(os.path.join(sim_dir, name))
    return obs


def _read_scalJ(name):
    """Returns dictionary containing anaysis results from scalar observable.
    """
    with open(name) as f:
        lines = f.readlines()
    N_obs = int((len(lines)-3)/2)

    sign = np.loadtxt(lines[-1].split()[-2:])

    obs = np.zeros([N_obs, 2])
    for iobs in range(N_obs):
        obs[iobs] = lines[iobs+2].split()[-2:]

    return {'sign': sign, 'obs': obs}


def _read_eqJ(name):
    """Returns dictionary containing anaysis results from equal time
    correlation function
    """
    with open(name) as f:
        lines = f.readlines()

    if name.endswith('K'):
        x_name = 'k'
    elif name.endswith('R'):
        x_name = 'r'

    N_lines = len(lines)
    for i in range(1, N_lines):
        if len(lines[i].split()) == 2:
            N_orb = int(np.sqrt(i-1))
            break

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
