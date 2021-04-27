[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)

## pyALF

The package implements ALF's python interface. `pyALF` greatly simplifies using the code, making it ideal for:

* *obtaining benchmark* results for established models;
* *getting started* with QMC and ALF;
* or just *quickly running* a simulation with ALF.


## Prerequisites

* Python
* Jupyter
* the libraries Lapack and Blas
* a Fortran compiler, such as gfortran or ifort,

where the last two are required by the main package [ALF](https://git.physik.uni-wuerzburg.de:ALF).

Also, add pyALF's path to your environment variable `PYTHONPATH`. In Linux, this can be achieved, e.g., by adding the following line to `.bashrc`:

```bash
export PYTHONPATH="/local/path/to/pyALF:$PYTHONPATH"
```


## Usage

Jupyter notebooks [are run](https://jupyter.readthedocs.io/en/latest/running.html) through a Jupyter server started, e.g., from the command line:

```bash
jupyter notebook
```
or
```bash
jupyter-notebook
```
which opens the "notebook dashboard" in your default browser, where you can navigate through your file structure to the pyALF directory. There you will find the interface's core module, `py_alf.py`, some auxiliary files, and a number of notebooks.

However, pyALF can also be used to start a simulation from the command line, without starting a Jupyter server. For instance:

```bash
python3.7 Run.py -R  --alfdir /home/debian/ALF-1.2/ --config "Intel"  --executable_R Hubbard --mpi True  &
```
Notice that `Run.py` assumes the existence of the configuration file `Sims`, which defines the simulation parameters. A line of `Sims` might read as:
```bash
{"Model": "Hubbard", "Lattice_type": "Square", "L1": 4 , "L2": 4, "NBin": 5, "ham_T": 0.0, "Nsweep" : 2000, "Beta": 1.0, "ham_chem": -1.0 }
```

## Files and directories

* `py_alf.py` - provides interfaces for compilig, running and postprocessing ALF in Python
* `default_variables.py` - defines dictionaries containing all ALF parameters with default values
* `Run.py` - helper script for compiling, running and testing ALF
* `Sims` - configuration for running directly from the command line
* `Notebooks` - directory containing Jupyter notebooks



## License
The various works that make up the ALF project are placed under licenses that put
a strong emphasis on the attribution of the original authors and the sharing of the contained knowledge.
To that end we have placed the ALF source code under the GPL version 3 license (see license.GPL and license.additional)
and took the liberty as per GPLv3 section 7 to include additional terms that deal with the attribution
of the original authors(see license.additional).
The Documentation of the ALF project by the ALF contributors is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (see Documentation/license.CCBYSA)
We mention that we link against parts of lapack which licensed under a BSD license(see license.BSD).

