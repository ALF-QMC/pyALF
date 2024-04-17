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

## Installation

---
**⚠️ NOTE** In previous versions of pyALF, the installation instructions asked the users to set the environment variable `PYTHONPATH`.
This conflicts with the newer pip package, therefore you should remove definitions of this environment variable related to pyALF.

---

pyALF can be installed via the Python package installer [pip](https://pip.pypa.io/en/stable/).

```bash
pip install pyALF
```

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
git clone https://git.physik.uni-wuerzburg.de/ALF/pyALF.git
cd pyALF
python -m venv .venv
source .venv/bin/activate

pip install --editable .
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

pyALF also delivers a set of command line scripts, to be use from a UNIX shell. For a full list of command line scripts see [here](gitpages.physik.uni-wuerzburg.de/Jonas_schwab/pyalf-docu/source/reference/cli.html).

Try, e.g.

```bash
alf_run -h
```

The source code for the scripts can be found in the folder `/py_alf/cli/`.


### Use module `py_alf` in custom scripts

Finally, one can also use the module module `py_alf` in custom Python scripts, which is analogous to the usage in Jupyter notebooks minus some interactivity.

## License

The various works that make up the ALF project are placed under licenses that put
a strong emphasis on the attribution of the original authors and the sharing of the contained knowledge.
To that end we have placed the ALF source code under the GPL version 3 license (see license.GPL and license.additional)
and took the liberty as per GPLv3 section 7 to include additional terms that deal with the attribution
of the original authors(see license.additional).
The Documentation of the ALF project by the ALF contributors is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License (see Documentation/license.CCBYSA)
We mention that we link against parts of lapack which licensed under a BSD license(see license.BSD).
