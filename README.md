[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)

## pyALF

A Python package building on top of [ALF](https://git.physik.uni-wuerzburg.de/ALF/ALF), meant to simplify the different steps of working with ALF, including:

* Obtaining and compiling the ALF source code
* Preparing and running simulations
* Postprocessing and displaying the data obtained during the simulation

It introduces:

* The Python module `py_alf`, exposing all the package's utility to Python.
* A set of command line tools in the folder `py_alf/cli`, that make it easy to leverage pyALF from a Unix shell.
* Jupyter notebooks in the folder `Notebooks`, serving as an easy introduction to QMC and ALF
* Python Scripts in the folder `Scripts` that can be run to reproduce benchmark results for established models

## Prerequisites

* Python3
* Jupyter
* The following Python packages:
  * h5py
  * numpy
  * pandas
  * matplotlib
  * numba
  * scipy
  * tkinter
  * ipywidgets
  * ipympl
  * f90nml
* the libraries Lapack and Blas
* a Fortran compiler, such as gfortran or ifort,

where the last two are required by the main package [ALF](https://git.physik.uni-wuerzburg.de/ALF).

Also, add pyALF's path to your environment variable `PYTHONPATH`. In Linux, this can be achieved, e.g., by adding the following line to `.bashrc`:

```bash
export PYTHONPATH="/local/path/to/pyALF:$PYTHONPATH"
```

## Usage

Jupyter notebooks [are run](https://jupyter.readthedocs.io/en/latest/running.html) through a Jupyter server started, e.g., from the command line:

```bash
jupyter-notebook
```

or

```bash
jupyter-lab
```

which opens the "notebook dashboard" in your default browser, where you can navigate through your file structure to the pyALF directory. There you will find the interface's core module, `py_alf.py`, some auxiliary files, and a number of notebooks.

However, pyALF can also be used to start a simulation from the command line, without starting a Jupyter server. For instance, check the help message:

```bash
export PATH="/path/to/pyALF/py_alf/cli:$PATH"
alf_run.py -h
```

## License

The various works that make up the ALF project are placed under licenses that put
a strong emphasis on the attribution of the original authors and the sharing of the contained knowledge.
To that end we have placed the ALF source code under the GPL version 3 license (see license.GPL and license.additional)
and took the liberty as per GPLv3 section 7 to include additional terms that deal with the attribution
of the original authors(see license.additional).
The Documentation of the ALF project by the ALF contributors is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (see Documentation/license.CCBYSA)
We mention that we link against parts of lapack which licensed under a BSD license(see license.BSD).
