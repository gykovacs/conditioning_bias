"""
Testing the lattice feature determination
"""

import numpy as np

from flipping_random_forest import lattice_features

def test_lattice_features():
    """
    Testing the lattice features function
    """

    X = np.array([[1, 0], [2, 1], [3, 4]])

    flags = lattice_features(X)

    assert np.all(flags == np.array([True, False]))
