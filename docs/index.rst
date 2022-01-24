.. pyALF documentation master file, created by
   sphinx-quickstart on Wed Jan 12 14:50:38 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyALF's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
=================================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


class ALF_source
----------------------------------
.. autoclass:: py_alf.ALF_source
   :members:

class Simulation
----------------------------------
.. autoclass:: py_alf.Simulation
   :members:


High-level analysis functions
----------------------------------
.. autofunction:: alf_ana.analysis.analysis
.. autofunction:: alf_ana.check_warmup.check_warmup
.. autofunction:: alf_ana.check_rebin.check_rebin


Scripts
----------------------------------
alf_postprocess.py::
   Script for postprocessing monte carlo bins.

   positional arguments:
   directories           Directories to analyze. If empty, analyzes all directories
                           containing file "data.h5" it can find.

   optional arguments:
   -h, --help            show this help message and exit
   --check_warmup, --warmup
                           Check warmup.
   --check_rebin, --rebin
                           Check rebinning for controlling autocorrelation.
   -l CHECK_LIST [CHECK_LIST ...], --check_list CHECK_LIST [CHECK_LIST ...]
                           List of observables to check for warmup and rebinning.
   --do_analysis, --ana  Do analysis.
   --gather              Gather all analysis results in one file.
   --no_tau              Skip time displaced correlations.
   --custom_obs CUSTOM_OBS
                           File that defines custom observables.
   --symmetry SYMMETRY, --sym SYMMETRY
                           File that defines lattice symmetries.


.. Module alf_ana.ana
Low-level analysis functions
----------------------------------
.. automodule:: alf_ana.ana
   :members:

Object alf_ana.lattice.Lattice
----------------------------------
.. autoclass:: alf_ana.lattice.Lattice
   :members:
