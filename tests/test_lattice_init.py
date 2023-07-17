#!/usr/bin/env python3
"""Testing the Python version of the lattice init vs the Fortran version."""

import numpy as np

from py_alf.lattice import _init0, _init1

lattice_types = {
    "Square": {
        'a1': np.array([1.0, 0.0]),
        'a2': np.array([0.0, 1.0]),
        },
    "Honeycomb": {
        'a1': np.array([1.0, 0.0]),
        'a2': np.array([0.5, np.sqrt(3)/2]),
        },
    "Pi_Flux": {
        'a1': np.array([1.0, 1.0]),
        'a2': np.array([1.0, -1.0]),
        },
}

def test_lattice_init(lattice_type, L1_1, L1_2, L2_1, L2_2):
    print('Testing', lattice_type, L1_1, L1_2, L2_1, L2_2)
    a1 = lattice_types[lattice_type]['a1']
    a2 = lattice_types[lattice_type]['a2']
    L1 = L1_1*a1 + L1_2*a2
    L2 = L2_1*a1 + L2_2*a2
    init0 = _init0(L1, L2, a1, a2)
    init1 = _init1(L1, L2, a1, a2)

    for a, b in zip(init0, init1):
        if not np.allclose(a, b):
            raise Exception('not matching', a, b)

for lattice_type, L1_1, L1_2, L2_1, L2_2 in [
    ("Square", 4, 0, 0, 4),
    ("Square", 4, 1, 0, 4),
    ("Square", 4, 0, 1, 4),
    ("Square", 4, 0, 0, 1),
    ("Honeycomb", 3, 0, 0, 3),
    ("Honeycomb", 3, 1, 0, 3),
    ("Pi_Flux", 4, 0, 0, 4),
    ("Pi_Flux", 4, 1, 0, 4),
]:
    test_lattice_init(lattice_type, L1_1, L1_2, L2_1, L2_2)
