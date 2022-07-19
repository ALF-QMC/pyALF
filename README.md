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

The **documentation** can be found [here](http://gitpages.physik.uni-wuerzburg.de/Jonas_schwab/pyalf-docu).

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
* The libraries Lapack and Blas
* A Fortran compiler, such as gfortran or ifort,

where the last two are required by the main package [ALF](https://git.physik.uni-wuerzburg.de/ALF).

Also, add pyALF's path to your environment variable `PYTHONPATH`. In Linux, this can be achieved, e.g., by adding the following line to `~/.bashrc` if the used shell if bash or `~/.zshrc`, if the shell is zsh:

```bash
export PYTHONPATH="/local/path/to/pyALF:$PYTHONPATH"
```

## Usage

There are multiple ways to use pyALF, which roughly breaks down into three approaches:
* Using Jupyter notebooks
* Using the command line interface
* Use the module `py_alf` in custom scripts

### Jupyter notebooks

A convenient way to use pyALF is through Jupyter notebooks. They [are run](https://jupyter.readthedocs.io/en/latest/running.html) through a Jupyter server started, e.g., from the command line:

```bash
jupyter-notebook
```

or

```bash
jupyter-lab
```

which opens the "notebook dashboard" in your default browser, from where one can open the sample notebooks in `Notebooks/` and create new notebooks.

### Command line interface

pyALF also delivers a set of command line scripts, located in the folder `/py_alf/cli/`, to be use from a UNIX shell. For convenient access, it makes sense to add the folder to the environment variable `PATH`:

```bash
export PATH="/path/to/pyALF/py_alf/cli:$PATH"
```

Then the scripts can simply be called by their names, try e.g. 

```bash
alf_run.py -h
```

For a full list of command line scripts see [here](gitpages.physik.uni-wuerzburg.de/Jonas_schwab/pyalf-docu/source/reference/cli.html)

### Use module `py_alf` in custom scripts

Finally, one can also use the module module `py_alf` in custon Python scripts, which is analogous to the usage in Jupyter notebooks minus some interactivity.

## License

The various works that make up the ALF project are placed under licenses that put
a strong emphasis on the attribution of the original authors and the sharing of the contained knowledge.
To that end we have placed the ALF source code under the GPL version 3 license (see license.GPL and license.additional)
and took the liberty as per GPLv3 section 7 to include additional terms that deal with the attribution
of the original authors(see license.additional).
The Documentation of the ALF project by the ALF contributors is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (see Documentation/license.CCBYSA)
We mention that we link against parts of lapack which licensed under a BSD license(see license.BSD).
