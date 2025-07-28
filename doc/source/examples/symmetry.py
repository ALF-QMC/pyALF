"""File that defines lattice symmetry for analysis."""

import numpy as np


# Define list of transformations (Lattice, i) -> new_i
# Default analysis will average over all listed elements
symmetry = [
    lambda latt, i : i,
    lambda latt, i : latt.rotate(i, np.pi*0.5),
    lambda latt, i : latt.rotate(i, np.pi),
    lambda latt, i : latt.rotate(i, np.pi*1.5),
]
