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


Class ALF_source
----------------------------------
.. autoclass:: py_alf.ALF_source
   :members:

Class Simulation
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
A number of executable python scripts in the folder `Scripts/`.
For productive work, it may be suitable to add this folder to
the `$PATH` environment variable.

minimal_ALF_run.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Extensively commented example script showing the minimal steps
for creating and running an ALF simulation in pyALF.

alf_run.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. argparse::
    :ref: Scripts.alf_run._get_arg_parser
    :prog: alf_run.py
    :nodefault:

alf_postprocess.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. argparse::
    :ref: Scripts.alf_postprocess._get_arg_parser
    :prog: alf_postprocess.py

alf_bin_count.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. argparse::
    :ref: Scripts.alf_bin_count._get_arg_parser
    :prog: alf_bin_count.py
    :nodefault:

alf_show_obs.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. argparse::
    :ref: Scripts.alf_show_obs._get_arg_parser
    :prog: alf_show_obs.py
    :nodefault:

alf_del_bins.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. argparse::
    :ref: Scripts.alf_del_bins._get_arg_parser
    :prog: alf_del_bins.py
    :nodefault:

alf_test_branch.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. argparse::
    :ref: Scripts.alf_test_branch._get_arg_parser
    :prog: alf_test_branch.py
    :nodefault:


.. Module alf_ana.ana

Low-level analysis functions
----------------------------------
.. automodule:: alf_ana.ana
   :members:

Class alf_ana.lattice.Lattice
----------------------------------
.. autoclass:: alf_ana.lattice.Lattice
   :members:
