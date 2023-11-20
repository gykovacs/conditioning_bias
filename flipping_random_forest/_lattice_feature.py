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
