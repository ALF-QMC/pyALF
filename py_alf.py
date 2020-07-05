"""Provides interfaces for compilig, running and postprocessing ALF in Python.
"""

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

    Provides functions for preparing, running and postprocessing a simulation.
    """

    def __init__(self, sim, alf_dir='ALF', sim_dir=None, executable=None,
                 compile_config='GNU noMPI', branch=None):
        self.sim = sim
        self.alf_dir = os.path.abspath(alf_dir)
        if sim_dir is None:
            self.sim_dir = os.path.abspath(directory_name(sim))
        else:
            self.sim_dir = os.path.abspath(sim_dir)

        if executable is None:
            executable = sim['Model']
        self.executable = executable
        self.compile_config = compile_config
        self.branch = branch

    def compile(self):
        compile_alf(self.alf_dir, self.branch, self.compile_config,
                    model='all')

    def run(self):
        print('Prepare directory {} for Monte Carlo run'
              .format(self.sim_dir))
        if not os.path.exists(self.sim_dir):
            os.mkdir(self.sim_dir)

        with cd(self.sim_dir):
            copyfile(
                os.path.join(self.alf_dir, 'Scripts_and_Parameters_files',
                             'Start', 'seeds'),
                'seeds')
            params = set_param(self.sim)
            write_parameters(params)
            out_to_in(verbose=False)
            executable = os.path.join(self.alf_dir, 'Prog',
                                      self.executable+'.out')
            print('Run {}'.format(executable))
            try:
                Run = subprocess.run(executable,
                                     check=True,
                                     stderr=subprocess.STDOUT,
                                     text=True,
                                     capture_output=True)
            except subprocess.CalledProcessError:
                print('Error while running {}.'.format(executable))
            print('Output:\n', Run.stdout)

    def ana(self):
        with cd(self.sim_dir):
            analysis(self.alf_dir)

    def get_obs(self, names=None):
        return get_obs(self.sim_dir, names)


def convert_par_to_str(parameter):
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

    raise Exception('Error in "convert_par_to_str": unrecognized type')


def write_parameters(params):
    with open('parameters', 'w') as file:
        for namespace in params:
            file.write("&{}\n".format(namespace))
            for var in params[namespace]:
                file.write(var + ' = '
                           + convert_par_to_str(params[namespace][var]) + '\n')
            file.write("/\n\n")


def directory_name(sim):
    dirname = ''
    for name in sim:
        if name in ["L1", "L2", "Lattice_type", "Model",
                    "Checkerboard", "Symm", "N_SUN", "N_FL", "Phi_X", "N_Phi",
                    "Dtau", "Beta", "Projector",
                    "Theta", "ham_T", "ham_chem", "ham_U",
                    "ham_T2", "ham_U2", "ham_Tperp"]:
            if name in ["Lattice_type", "Model"]:
                dirname = '{}{}_'.format(dirname, sim[name])
            else:
                dirname = '{}{}={}_'.format(
                    dirname, name.strip("ham_"), sim[name])
    return dirname[:-1]


def update_var(params, var, value):
    """Try to update value of parameter called var in params."""
    for name in params:
        for var2 in params[name]:
            if var2 == var:
                params[name][var2] = value
                return params
    raise Exception('"{}" does not correspond to a parameter'.format(var))


def set_param(sim):
    model = sim['Model']
    params = default_variables.PARAMS
    params_model = default_variables.PARAMS_MODEL
    params['VAR_'+model] = params_model['VAR_'+model]

    for var in sim:
        params = update_var(params, var, sim[var])
    return params


def compile_alf(alf_dir='ALF', branch=None, config='GNU noMPI', model='all',
                url='git@git.physik.uni-wuerzburg.de:ALF/ALF.git'):
    """Compiles ALF. Clones a new repository if alf_dir does not exist."""

    alf_dir = os.path.abspath(alf_dir)
    if not os.path.exists(alf_dir):
        print("Repository {} does not exist, cloning from {}"
              .format(alf_dir, url))
        try:
            Run = subprocess.run(["git", "clone", url, alf_dir],
                                 check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            print('Error while cloning repository')
            print(Run.stdout)

    with cd(alf_dir):
        if branch is not None:
            try:
                Run = subprocess.run(["git", "checkout", branch],
                                     check=True, capture_output=True,
                                     text=True)
            except subprocess.CalledProcessError:
                print('Error while checking out {}'.format(branch))
                print(Run.stdout)
        os.system(". ./configureHPC.sh " + config + "; make clean " + model)


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


def analysis(alfdir):
    """Perform the default analysis on all files ending in _scal, _eq or _tau
    in current working directory.
    """
    if os.path.exists('Var_scal'):
        os.remove('Var_scal')
    for name in os.listdir():
        if name.endswith('_scal'):
            print('Analysing {}'.format(name))
            os.symlink(name, 'Var_scal')
            command = alfdir + '/Analysis/cov_scal.out'
            os.system(command)
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
            command = alfdir + '/Analysis/cov_eq.out'
            os.system(command)
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
            command = alfdir + '/Analysis/cov_tau.out'
            os.system(command)
            os.remove('intau')
            os.replace('SuscepJ', name+'JK')

            for name2 in os.listdir():
                if name2.startswith('g_'):
                    directory = name[:-4] + name2[1:]
                    if not os.path.exists(directory):
                        os.mkdir(directory)
                    os.replace(name2, os.path.join(directory, name2))


def get_obs(sim_dir, names=None):
    obs = {}
    if names is None:
        names = os.listdir(sim_dir)
    for name in names:
        if name.endswith('_scalJ'):
            obs[name] = read_scalJ(os.path.join(sim_dir, name))
        if name.endswith('_eqJK') or name.endswith('_eqJR'):
            obs[name] = read_eqJ(os.path.join(sim_dir, name))
    return obs


def read_scalJ(name):
    with open(name) as f:
        lines = f.readlines()
    N_obs = int((len(lines)-3)/2)

    sign = np.loadtxt(lines[-1].split()[-2:])

    obs = np.zeros([N_obs, 2])
    for iobs in range(N_obs):
        obs[iobs] = lines[iobs+2].split()[-2:]

    return {'sign': sign, 'obs': obs}


def read_eqJ(name):
    with open(name) as f:
        lines = f.readlines()

    if name[-1] == 'K':
        x_name = 'k'
    elif name[-1] == 'R':
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
