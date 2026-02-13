# Introduction

The auxiliary-field quantum Monte Carlo package ALF {cite}`ALF_v1,ALF_v2` is a powerful tool for simulating a broad set of fermionic systems, but since it is written in Fortran, it is not very dynamic and can be a bit daunting for new users.

Aiming to address this challenge, pyALF is a set of Python scripts built on top of ALF. It is meant to simplify the different steps of working with ALF, including:

- Obtaining and compiling the ALF source code
- Preparing and running simulations
- Postprocessing and displaying the data obtained during the simulation 

The source codes for both ALF and pyALF are publicly available at [https://github.com/ALF-QMC](https://github.com/ALF-QMC).

This documentation is structured in the following way:

1. {numref}`sec:pyalf_install` describes the prerequisites of pyALF and how to set things up to be able to use it in a productive manner.
2. {numref}`sec:pyalf_usage` displays the features of pyALF and how to use them on small examples.
3. For a reference on pyALF's features, see {numref}`sec:pyalf_reference`.
