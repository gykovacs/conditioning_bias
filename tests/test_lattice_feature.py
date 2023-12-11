"""
Testing the lattice feature determination
"""

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

from conditioning_bias import lattice_features, count_lattice_splits


def test_lattice_features():
    """
    Testing the lattice features function
    """

    X = np.array([[1, 0], [2, 1], [3, 4]])

    flags = lattice_features(X)

    assert np.all(flags == np.array([True, False]))


def test_count_lattice_splits():
    """
    Testing the counting of lattice splits
    """

    dataset = load_iris()
    X = dataset["data"]
    y = dataset["target"]

    random_state = np.random.RandomState(5)

    X[:, 1] = X[:, 1] + random_state.random_sample(X.shape[0])

    dt = DecisionTreeClassifier(random_state=5).fit(X, y)

    n_lattice, n_splits = count_lattice_splits(X, dt)
    assert n_lattice <= n_splits
