"""Define C_4 symmetry (=fourfold rotation) for pyALF analysis."""
from math import pi

# Define list of transformations (Lattice, i) -> new_i
# Default analysis will average over all listed elements
symmetry = [
    lambda latt, i : i,
    lambda latt, i : latt.rotate(i, pi*0.5),
    lambda latt, i : latt.rotate(i, pi),
    lambda latt, i : latt.rotate(i, pi*1.5),
]
