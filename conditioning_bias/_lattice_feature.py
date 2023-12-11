"""
This module implements the lattice feature test
"""

import numpy as np

def lattice_feature(x: np.array, threshold=1) -> bool:
    """
    Determines if a feature is lattice feature

    Args:
        x (np.array): the feature values
        threshold (int|str): the threshold on the count of lattice
                            costellations or 'sqrt' for a dynamic
                            threshold based on the square root of the
                            number of unique feature values

    Returns:
        bool: True indicates a lattice feature
    """
    x_u = np.unique(x)

    threshold = threshold if threshold != "sqrt" else np.sqrt(x_u.shape[0])

    for x_val in x_u:
        x_diff = np.round(np.abs(x_u - x_val), 12)
        has = np.any(np.unique(x_diff, return_counts=True)[1] > threshold)
        if has:
            break

    return bool(has)

def lattice_features(X: np.array, threshold=1) -> bool:
    """
    Determines if the features are lattice features

    Args:
        X (np.array): the feature vectors

    threshold (int|str): the threshold on the count of lattice
                            costellations or 'sqrt' for a dynamic
                            threshold based on the square root of the
                            number of unique feature values

    Returns:
        np.array: an array of flags, True indicates a lattice feature
    """

    return np.array([lattice_feature(X[:, idx], threshold) for idx in range(X.shape[1])])

def count_lattice_splits(X: np.array, tree):
    """
    Count the lattice splits in a particular tree

    Args:
        X (np.array): the training set
        tree (obj): the tree to calculate the lattice splits in

    Returns:
        int, int: the number of lattice splits and total splits
    """
    lattice = lattice_features(X)
    threshold = tree.tree_.threshold
    feature = tree.tree_.feature

    n_splits = np.sum(tree.tree_.children_left != tree.tree_.children_right)

    n_lattice_splits = 0
    for idx, lflag in enumerate(lattice):
        if not lflag:
            continue

        thresholds = threshold[feature == idx]

        for th in thresholds:
            diffs = np.round(np.abs(np.unique(X[:, idx]) - th), 8)
            n_lattice_splits += np.any(np.unique(diffs, return_counts=True)[1] > 1) and 0.0 in diffs

    return n_lattice_splits, n_splits
