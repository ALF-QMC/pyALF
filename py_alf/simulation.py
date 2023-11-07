"""
Provides interfaces for compiling, running and postprocessing ALF in Python.
"""
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes

__author__ = "Jonas Schwab"
__copyright__ = "Copyright 2020-2022, The ALF Project"
__license__ = "GPL"

import os
import re
import subprocess
import shutil
import sys

import numpy as np
import pandas as pd
from . analysis import analysis
from . ana import load_res
from . alf_source import ALF_source


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
    """
    Object corresponding to an ALF simulation.

    Parameters
    ----------
    alf_src : ALF_source
        Objet representing ALF source code.
    ham_name : str
        Name of the Hamiltonian.
    sim_dict : dict or list of dicts
        Dictionary specfying parameters owerwriting defaults.
        Can be a list of dictionaries to enable parallel tempering.
    sim_dir : path-like object, optional
        Directory in which the Monte Carlo will be run.
        If not specified, sim_dir is generated from sim_dict.
    sim_root : path-like object, default="ALF_data"
        Directory to prepend to sim_dir.
    mpi : bool, default=False
        Employ MPI.
    parallel_params : bool, default=False
        Run independent parameter sets in parallel.
        Based on parallel tempering, but without exchange steps.
    n_mpi : int, default=2
        Number of MPI processes if mpi is true.
    n_omp : int, default=1
        Number of OpenMP threads per process.
    mpiexec : str, default="mpiexec"
        Command used for starting a MPI run. This may have to be adapted to
        fit with the MPI library used at compilation. Possible candidates
        include 'orterun', 'mpiexec.hydra'.
    mpiexec_args : list of str, optional
        Additional arguments to MPI executable. E.g. the flag
        ``--hostfile /path/to/file`` is specified by
        ``mpiexec_args=['--hostfile', '/path/to/file']``
    machine : {"GNU", "INTEL", "PGI", "Other machines defined in configure.sh"}
        Compiler and environment.
    stab : str, optional
        Stabilization strategy employed by ALF.
        Possible values: "STAB1", "STAB2", "STAB3", "LOG". Not case sensitive.
    devel : bool, default=False
        Compile with additional flags for development and debugging.
    hdf5 : bool, default=True
        Whether to compile ALF with HDF5.
        Full postprocessing support only exists with HDF5.
    """
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    def __init__(self, alf_src, ham_name, sim_dict, **kwargs):
        if not isinstance(alf_src, ALF_source):
            raise TypeError('alf_src needs to be an instance of ALF_source')
        self.alf_src = alf_src
        self.ham_name = ham_name
        self.sim_dict = sim_dict
        self.sim_dir = os.path.abspath(os.path.expanduser(os.path.join(
            kwargs.pop("sim_root", "ALF_data"),
            kwargs.pop("sim_dir",
                       directory_name(alf_src, ham_name, sim_dict)))))
        self.mpi = kwargs.pop("mpi", False)
        self.parallel_params = kwargs.pop("parallel_params", False)
        self.n_mpi = kwargs.pop("n_mpi", 2)
        self.n_omp = kwargs.pop('n_omp', 1)
        self.mpiexec = kwargs.pop('mpiexec', 'mpiexec')
        self.mpiexec_args = kwargs.pop('mpiexec_args', [])
        if not isinstance(self.mpiexec_args, list):
            raise TypeError('mpiexec_args has to be a list.')
        stab = kwargs.pop('stab', '').upper()
        machine = kwargs.pop('machine', 'GNU').upper()
        self.devel = kwargs.pop('devel', False)
        self.hdf5 = kwargs.pop('hdf5', True)
        if kwargs:
            raise TypeError('Unused keyword arguments: {}'.format(kwargs))

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
                        raise TypeError(
                            'Parameter {} not listed in default_variables'
                            .format(par_name))
        else:
            for par_name in self.sim_dict:
                if par_name.upper() not in p_list:
                    raise TypeError(
                        'Parameter {} not listed in default_variables'
                        .format(par_name))

        if self.mpi and self.n_mpi is None:
            raise TypeError('You have to specify n_mpi if you use MPI.')

        if self.parallel_params and (not self.tempering):
            raise TypeError('sim_dict has to be a list to use Parallel parameters feature.')

        if stab not in ['STAB1', 'STAB2', 'STAB3', 'LOG', '']:
            raise TypeError('Illegal value stab={}'.format(stab))

        self.config = '{} {}'.format(machine, stab).strip()

        if self.mpi:
            if self.parallel_params:
                self.config += ' PARALLEL_PARAMS'
            elif self.tempering:
                self.config += ' TEMPERING'
            else:
                self.config += ' MPI'
        else:
            self.config += ' NOMPI'

        if self.devel:
            self.config += ' DEVEL'

        if self.hdf5:
            self.config += ' HDF5'

        self.config += ' NO-INTERACTIVE'

    def compile(self, verbosity=0):
        """
        Compile ALF.

        Parameters
        ----------
        verbosity : int, default=0
            0: Don't echo make reciepes.
            1: Echo make reciepes.
            else: Print make tracing information.
        """
        compile_alf(self.alf_src.alf_dir, config=self.config,
                    verbosity=verbosity, branch=self.alf_src.branch)

    def run(self, copy_bin=False, only_prep=False, bin_in_sim_dir=False):
        """
        Prepare simulation directory and run ALF.

        Parameters
        ----------
        copy_bin : bool, default=False
            Copy ALF binary into simulation directory.
        only_prep : bool, default=False
            Do not run ALF, only prepare simulation directory.
        bin_in_sim_dir : bool, default=False
            Assume that the ALF binary is already present in simultation
            directory and use this.
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

        executable = os.path.join(self.alf_src.alf_dir, 'Prog', 'ALF.out')
        if copy_bin:
            shutil.copy(executable, self.sim_dir)
            executable = os.path.join(self.sim_dir, 'ALF.out')
        if bin_in_sim_dir:
            executable = os.path.join(self.sim_dir, 'ALF.out')
        if only_prep:
            return
        env = getenv(self.config, self.alf_src.alf_dir)
        env['OMP_NUM_THREADS'] = str(self.n_omp)
        with cd(self.sim_dir):
            print('Run {}'.format(executable))
            try:
                if self.mpi:
                    command = [self.mpiexec, '-n', str(self.n_mpi),
                               *self.mpiexec_args, executable]
                else:
                    command = executable
                subprocess.run(command, check=True, env=env)
            except subprocess.CalledProcessError as ALF_crash:
                print('Error while running {}.'.format(executable))
                print('parameters:')
                # with open('parameters', 'r') as f:
                #     print(f.read())
                raise RuntimeError('Error while running {}.'.format(executable)) \
                    from ALF_crash

    def get_directories(self):
        """Return list of directories connected to this simulation."""
        if self.tempering:
            directories = [os.path.join(self.sim_dir, "Temp_{}".format(i))
                           for i in range(len(self.sim_dict))]
        else:
            directories = [self.sim_dir]
        return directories

    def print_info_file(self):
        """Print info file(s) that get generated by ALF."""
        for directory in self.get_directories():
            filename = os.path.join(directory, 'info')
            if os.path.exists(filename):
                print('===== {} ====='.format(filename))
                with open(filename, 'r', encoding='UTF-8') as f:
                    print(f.read())
            else:
                print('{} does not exist.'.format(filename))
                return

    # pylint: disable-next=inconsistent-return-statements
    def check_warmup(self, names, gui='tk', **kwargs):
        """
        Plot bins to determine n_skip.

        Parameters
        ----------
        names : list of str
            Names of observables to check.
        gui : {'tk', 'ipy'}
            Whether to use Tkinter or Jupyter Widget for GUI. default: 'tk'
        **kwargs : dict, optional
            Extra arguments for :func:`py_alf.check_warmup_tk` or
            :func:`py_alf.check_warmup_ipy`.
        """
        if gui == 'tk':
            # pylint: disable-next=import-outside-toplevel
            from . check_warmup_tk import check_warmup_tk
            check_warmup_tk(self.get_directories(), names, **kwargs)
        elif gui == 'ipy':
            # pylint: disable-next=import-outside-toplevel
            from . check_warmup_ipy import check_warmup_ipy
            return check_warmup_ipy(self.get_directories(), names, **kwargs)
        else:
            raise TypeError(f'Illegal value gui={gui}')

    # pylint: disable-next=inconsistent-return-statements
    def check_rebin(self, names, gui='tk', **kwargs):
        """
        Plot error vs n_rebin to control autocorrelation.

        Parameters
        ----------
        names : list of str
            Names of observables to check.
        gui : {'tk', 'ipy'}
            Whether to use Tkinter or Jupyter Widget for GUI. default: 'tk'
        **kwargs : dict, optional
            Extra arguments for :func:`py_alf.check_rebin_tk` or
            :func:`py_alf.check_rebin_ipy`.
        """
        if gui == 'tk':
            # pylint: disable-next=import-outside-toplevel
            from . check_rebin_tk import check_rebin_tk
            check_rebin_tk(self.get_directories(), names, **kwargs)
        elif gui == 'ipy':
            # pylint: disable-next=import-outside-toplevel
            from . check_rebin_ipy import check_rebin_ipy
            return check_rebin_ipy(self.get_directories(), names, **kwargs)
        else:
            raise TypeError(f'Illegal value gui={gui}')

    def analysis(self, python_version=True, **kwargs):
        """
        Perform default analysis on Monte Carlo data.

        Calls :func:`py_alf.analysis`, if run with `python_version=True`.

        Parameters
        ----------
        python_version : bool, default=True
            Use Python version of analysis.
            The non-Python version is legacy and does not support all
            postprocessing features.
        **kwargs : dict, optional
            Extra arguments for :func:`py_alf.analysis`, if run with
            `python_version=True`.
        """
        for directory in self.get_directories():
            if python_version:
                analysis(directory, **kwargs)
            else:
                analysis_fortran(self.alf_src.alf_dir, directory,
                                 hdf5=self.hdf5)

    def get_obs(self, python_version=True):
        """Return Pandas DataFrame containing anaysis results from observables.

        The non-python version is legacy and does not support all
        postprocessing features, e.g. time-displaced observables.
        """
        if python_version:
            return load_res(self.get_directories())

        dicts = {}
        for directory in self.get_directories():
            dicts[directory] = get_obs(directory, names=None)
        return pd.DataFrame(dicts).transpose()


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
    """Convert a given parameter value to a string for parameter file."""
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

    raise TypeError('Error in "_convert_par_to_str": unrecognized type')


def write_parameters(params):
    """Write nameslists to file 'parameters'."""
    with open('parameters', 'w', encoding='UTF-8') as file:
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
    """Return name of directory for simulations, given a set of simulation
    parameters.
    """
    p_list = alf_src.get_params_names(ham_name, include_generic=False)
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
    raise TypeError('"{}" does not correspond to a parameter'.format(var))


def set_param(alf_src, ham_name, sim_dict):
    """Return dictionary containing all parameters needed by ALF.

    Input: Dictionary with chosen set of <parameter: value> pairs.
    Output: Dictionary containing all namelists needed by ALF.
    """
    params = alf_src.get_default_params(ham_name)

    params["VAR_ham_name"] = {
        "ham_name": {'value': ham_name, 'comment': "Name of Hamiltonian"}
    }
    params.move_to_end('VAR_ham_name', last=False)

    for name, value in sim_dict.items():
        params = _update_var(params, name, value)
    return params


def getenv(config, alf_dir='.'):
    """Get environment variables for compiling ALF."""
    with cd(alf_dir):
        try:
            subprocess.run(
                ['bash', '-c',
                '. ./configure.sh {} NO-FALLBACK > /dev/null || exit 1 &&'
                'env > environment'.format(config)],
                check=True)
        except subprocess.CalledProcessError:
            print(f'Error while running configure.sh with "{config}"\n'
                  'Is your machine set corretly?', file=sys.stderr)
        with open('environment', 'r', encoding='UTF-8') as f:
            lines = f.readlines()
    env = {}
    for line in lines:
        if ((not re.search(r"^BASH_FUNC.*%%=()", line))
            and '=' in line
            and line[0] != ' '):
            item = line.strip().split("=", 1)
            if len(item) == 2:
                env[item[0]] = item[1]
            else:
                env[item[0]] = ''
    return env


def compile_alf(alf_dir=os.getenv('ALF_DIR', './ALF'),
                branch=None,
                config='GNU noMPI',
                url='https://git.physik.uni-wuerzburg.de/ALF/ALF.git',
                verbosity=0
                ):
    """
    Compile ALF. Clone a new repository if alf_dir does not exist.

    Parameters
    ----------
    alf_dir : path-like object, optional
        Directory containing the ALF source code. If the directory does
        not exist, the source code will be fetched from a server.
        Defaults to environment variable $ALF_DIR if present, otherwise
        to './ALF'.
    branch : str, optional
        If specified, this will be checked out by git.
    config : str, default='GNU noMPI'
        Arguments for `configure.sh`.
    url : str, default='https://git.physik.uni-wuerzburg.de/ALF/ALF.git'
        Address from where to clone ALF if alf_dir not exists.
    verbosity : int, default=0
        0: Don't echo make reciepes.
        1: Echo make reciepes.
        else: Print make tracing information.
    """
    if verbosity == 0:
        makeflags = ['-s']
    elif verbosity == 1:
        makeflags = []
    else:
        makeflags = ['--trace']

    alf_dir = os.path.abspath(alf_dir)
    if not os.path.exists(alf_dir):
        print("Repository {} does not exist, cloning from {}"
              .format(alf_dir, url))
        try:
            subprocess.run(["git", "clone", url, alf_dir], check=True)
        except subprocess.CalledProcessError as git_clone_failed:
            raise RuntimeError('Error while cloning repository') \
                from git_clone_failed

    with cd(alf_dir):
        if branch is not None:
            print('Checking out branch {}'.format(branch))
            try:
                subprocess.run(['git', 'checkout', branch], check=True)
            except subprocess.CalledProcessError as git_checkout_failed:
                raise RuntimeError('Error while checking out {}'.format(branch)) \
                    from git_checkout_failed
        env = getenv(config)
        print('Compiling ALF... ')
        subprocess.run(['make', *makeflags, 'clean'], check=True, env=env)
        subprocess.run(['make', *makeflags, 'all'], check=True, env=env)
        print('Done.')


def out_to_in(verbose=False):
    """Rename all output configurations confout_* to confin_*.

    For continuing the Monte Carlo simulation where the previous stopped.
    """
    for name in os.listdir():
        if name.startswith('confout_'):
            name2 = 'confin_' + name[8:]
            if verbose:
                print('mv {} {}'.format(name, name2))
            os.replace(name, name2)


################ Legacy ################


def analysis_fortran(alf_dir, sim_dir='.', hdf5=False):
    """Perform the default analysis unsing ALFs own analysis routines
    on all files ending in _scal, _eq or _tau in directory sim_dir. Not fully
    supported.
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
    """Return dictionary with analysis results from Fortran analysis (legacy).

    Only scalar observables and equal time correlators.
    If names is None: gets all observables, else the ones listed in names.
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
                del temp2
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
    """Return dictionary with Fortran analysis
    results from scalar observable. (legacy)
    """
    with open(name, encoding='UTF-8') as f:
        lines = f.readlines()
    N_obs = int((len(lines)-2)/2)

    sign = np.loadtxt(lines[-1].split()[-2:])
    print(name, N_obs)

    obs = np.zeros([N_obs, 2])
    for iobs in range(N_obs):
        obs[iobs] = lines[2*iobs+2].split()[-2:]

    return {'sign': sign, 'obs': obs}


def _read_eqJ(name):
    """Return dictionary with Fortran analysis results from equal time
    correlation function. (legacy)
    """
    with open(name, encoding='UTF-8') as f:
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
