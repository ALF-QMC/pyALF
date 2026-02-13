(sec:pyalf_install)=
# Prerequisites and installation
$\phantom{\xi}$

This section lists the prerequisites of pyALF and how to set things up to be able to use it in a productive manner.

## ALF prerequisites

Since pyALF builds on ALF, we also want to satisfy its requirements. Note, however, that pyALF's postprocessing features are independent from ALF. This might be relevant, for example, when performing QMC runs and analysis on different machines.

The minimal ALF prerequisites are:

- The Unix shell Bash
- Make
- A recent Fortran Compiler (e. g. Submodules must be supported)
- BLAS+LAPACK
- Python 3

For parallelization, an MPI development library, e. g. Open MPI, is necessary.

Results from ALF can either be saved in a plain text format or HDF5, but full pyALF support is only provided for the latter, which is why in pyALF, HDF5 is enabled by default. ALF automatically downloads and compiles HDF5. For this to succeed, the following is needed:

- A C compiler (which is most often automatically included when installing a Fortran Compiler)
- A C++ preprocessor
- Curl or Wget
- gzip development libraries

The recommended way for obtaining the source code is through git.

Finally, the ALF testsuite needs:

- CMake

As an example, the requirements mentioned above can be satisfied on a Debian, Ubuntu, or similar operating system using the APT package manager, by executing the command:

```bash
sudo apt install make gfortran libblas-dev liblapack-dev \
           python3 libopenmpi-dev g++ curl libghc-zlib-dev \
           git ca-certificates cmake bash
```

The above installs compilers from the [GNU compiler collection](https://gcc.gnu.org/). Other supported and tested compiler frameworks are from the [Intel® oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html)
and the [NVIDIA HPC SDK](https://developer.nvidia.com/nvidia-hpc-sdk-downloads). The latter is denoted as `PGI` in ALF.

## pyALF installation


```{warning}
In previous versions of pyALF, the installation instructions asked the users to set the environment variable `PYTHONPATH`.
This conflicts with the newer pip package, therefore you should remove definitions of the `PYTHONPATH` environment variable related to pyALF.
```

pyALF can be installed via the Python package installer [pip](https://pip.pypa.io/en/stable/).

```bash
pip install pyALF
```

It automatically installs all requirements, but in case you want to install them in a 
different way, e.g. through apt or conda, these are the Python packages pyALF depends on:

- f90nml
- h5py
- ipympl
- ipywidgets
- matplotlib
- numba
- numpy
- pandas
- scipy
- tkinter

### Development installation

If you want to develop pyALF, you can clone the repository and install it in
[development mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html),
which allows you to edit the files while using them like an installed package.
For this, it is highly recommended to use a dedicated Python environment using e.g.
[Python venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
or a
[conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
The following example shows how to install pyALF in development mode using venv.

```bash
git clone https://https://github.com/ALF-QMC/pyALF.git
cd pyALF
python -m venv .venv
source .venv/bin/activate

pip install --editable .
```

## Setting ALF directory through environment variable

Since pyALF is set up to automatically clone ALF with git, it is not strictly necessary to download ALF manually, but pyALF will download ALF every time it does not find it. Therefore it is recommended to clone ALF once manually from [here](https://github.com/ALF-QMC/ALF) and setting its location in the environment variable `ALF_DIR`. This way, pyALF will use the same ALF source code directory every time.

ALF can be cloned with the Unix shell command

```bash
git clone https://github.com/ALF-QMC/ALF.git
```

This will create a folder called `ALF` in the current working directory of the terminal and download the repository there[^clone_fyi].

[^clone_fyi]: It is a lesser known fact that git is completely decentralized and the concept of a central repository is rather only a convention. Every git repository is an autonomous repository of itself. If, for example, pyALF has been cloned to `/path/to/ALF`, one could clone this repository with `git clone /path/to/ALF`.

The environment variable can then be set with the command

```bash
export ALF_DIR="/path/to/ALF"
```

where `/path/to/ALF` is the location of the ALF code, for example `/home/jonas/Programs/ALF`. To not have to repeat this command in every terminal session, it is advisable to add it to a file sourced when starting the shell, e.g. `~/.bashrc` or `~/.zshrc`.


## Check setup

To check if most things have been set up correctly, the script `minimal_ALF_run` can be used. It executes the same commands as the {doc}`usage/minimal_example`.
One should therefore be able to run it by executing 

```bash
minimal_ALF_run
```

in the Unix shell. If it does clone the ALF repository, `ALF_DIR` has not been set up correctly. Note that on the first compilation, ALF downloads and compiles HDF5, which can take up to ~15 minutes.

## Using Jupyter Notebooks

A convenient way to work with pyALF (and Python in general) is through Jupyter Notebooks. These are interactively usable documents that combine source code,
results and narration (through [Markdown](https://www.markdownguide.org/)) in one file. pyALF includes example notebooks, online available from [here](https://github.com/ALF-QMC/pyALF/tree/master/Notebooks), or by cloning the [pyALF repository](https://github.com/ALF-QMC/pyALF).

The canonical way to use the Jupyter Notebooks, is through a JupyterLab, which can for example be installed via pip (for more details see [here](https://jupyter.org/install)):

```bash
pip install jupyterlab
```

A JupyterLab can then be started with the shell command `jupyter-lab`, which launches a web server that should be automatically opened in your default browser.

Another convenient way to work with the notebooks is with [Visual Studio Code](https://code.visualstudio.com/), a versatile and extendable source-code editor.


## Ready-to-use container image

For a ready-to-use environment, one can use the Docker image [alfcollaboration/jupyter-pyalf-full](https://hub.docker.com/r/alfcollaboration/jupyter-pyalf-full), which has the above mentioned dependencies, ALF and pyALF installed. With a suitable container runtime e.g. [Docker](https://www.docker.com/) or [Podman](https://podman.io/), it can be used to run ALF and pyALF without any further setup. It is derived from the Jupyter Docker Stacks, therefore [this documentation](https://jupyter-docker-stacks.readthedocs.io) applies. For example, one could run a container like this:

```bash
docker run -it --rm -p 127.0.0.1:8888:8888 -v "$PWD":/home/jovyan/work \
    docker.io/alfcollaboration/jupyter-pyalf-full
```

- The [-p](https://docs.docker.com/reference/cli/docker/container/run/#publish) flag is used to expose port 8888
  and you can access a JupyterLab running within the container by navigating to `http://localhost:8888/lab?token=<token>`
  with you browser, where `<token>` has to be replaced by the token echoed to the terminal on startup.
- The [-v](https://docs.docker.com/reference/cli/docker/container/run/#volume) flag mounts the
  current working directory to `/home/jovyan/work` within the container, allowing to work on
  the same data in- and outside of the container.
- The [--rm](https://docs.docker.com/reference/cli/docker/container/run/#rm) flag instructs Docker
  to automatically remove the container after it exits, avoiding cluttering up the system with unused containers.
- The [-i](https://docs.docker.com/reference/cli/docker/container/run/#interactive) and
  [-t](https://docs.docker.com/reference/cli/docker/container/run/#tty) flags keep the
  container's `STDIN` open and attach a pseudo-terminal, allowing interactive input on the terminal.

It is also possible to use the container without launching the included JupyterLab. The following command
launches a container, which executes `minimal_ALF_run`, saving the results in the current working directory
and removing the container right after that.

```bash
docker run -it --rm -v "$PWD":/home/jovyan/work \
    docker.io/alfcollaboration/jupyter-pyalf-full \
    bash -c 'cd /home/jovyan/work && minimal_ALF_run'
```

## Some SSH port forwarding applications

ALF simulations are often performed on remote clusters that are accessed via SSH.
Notably, SSH can be used for much more than running a remote shell.
In this section, I will show how one can use SSH port forwarding to
download data to HPC clusters with restrictive firewalls and how
to access a JupyterLab launched on an HPC cluster.

### Use remote forwarding to circumvent restrictive firewalls

If one wanted to git clone the ALF source code, this could usually be
done with one of the following commands, using HTTPS or SSH, respectively.

```bash
git clone https://github.com/ALF-QMC/ALF.git
git clone git@github.com:ALF-QMC/ALF.git
```

But on some systems with very restrictive firewalls, this approach might
not work. This is where the ssh option [-R](https://man.openbsd.org/ssh#R)
might come in handy. It maps a port on the remote machine to a an address
connected to from the local machine on which the SSH command was executed.
To facilitate a connection to `github.com`, the following
commands can be used, connecting to port 443 or 22, for the HTTPS or SSH
protocol, respectively.

```bash
ssh -R <PortNum>:github.com:443 <username>@<servername>
ssh -R <PortNum>:github.com:22 <username>@<servername>
```

Here `<PortNum>` refers to a port on the remote machine, a value in the range
from 49152 to 65535 would be best here&nbsp;{cite}`rfc6335`.
And `<username>@<servername>` is the usual SSH address.
Alternatively to the command line option `-R`, the SSH config file option
[RemoteForward](https://man.openbsd.org/ssh_config#RemoteForward) can be used.

With these port forwarding options, the ALF source code can then be cloned on the remote
machine with:

```bash
git clone -c http.sslVerify=false https://localhost:<PortNum>/ALF-QMC/ALF.git
git clone ssh://git@localhost:<PortNum>/ALF-QMC/ALF.git
```

The HTTPS version needs the option `-c http.sslVerify=false` because the SSL certificate
for `github.com` does not apply to `localhost`.

One can omit the host value in the `-R` option (in the example above `github.com:443`)
which will set up a dynamic SOCKS proxy, able to connect to arbitrary addresses.
This can be used, for example, to download and install packages with `pip`.

```{warning}
Ports on the remote machine opened with `-R` / `RemoteForward` can not only be used
by you, but possibly also by other users of the machine. Therefore one should be
careful when using the options, in particular without specifying a host.
```

Using `-R` without a host to install pyALF with pip:

```bash
ssh -R <PortNum> <username>@<servername>
```

That pip can use the SOCKS proxy, the python package `pysocks` is necessary.
If the package is not yet available, it is enough to get the file 
`socks.py` from [here](https://github.com/Anorov/PySocks/blob/master/socks.py)
and have Python find it, e.g. with the environment variable `PYTHONPATH`.

Then pyALF can be installed with:

```bash
pip install --proxy socks4://localhost:<PortNum> pyALF
```


### Using Jupyter via SSH tunnel

When launching JupyterLab, it sets up a webserver and prints out how to access it locally, like:

```bash
http://localhost:<remote_port_number>/lab?token=<token>
```

Where `<remote_port_number>` is some port number (default 8888) and `<token>` is the password to access the server.

Now, to access this web server on the remote machine, one can forward this port to the local machine using the SSH option [-L](https://man.openbsd.org/ssh#L) and open it with the browser.

```bash
ssh -L <local_port_number>:localhost:<remote_port_number> <username>@<servername>
```

With the command from above, a remote JupyterLab will be accessible through the address `http://localhost:<local_port_number>:/lab?token=<token>`.


### Using SSH in Visual Studio Code

Here, a reference to use ssh in Visual Studio Code is provided: https://code.visualstudio.com/docs/remote/ssh
