"""
Testing the tree inference methods
"""

import numpy as np

from flipping_random_forest import tree_inference

def test_tree_inference():
    """
    Testing the tree inference
    """

    feature = np.array([0, -1, -1])
    threshold = np.array([1, -1, -1])
    value = np.array([[5], [6], [3]])
    children_left = np.array([1, -1, -1])
    children_right = np.array([2, -1, -1])

    result = tree_inference(
        X=np.array([[1]]),
        feature=feature,
        threshold=threshold,
        value=value,
        children_left=children_left,
        children_right=children_right,
        operator='<'
    )

    assert result[0][0][0] == 3.0

    result = tree_inference(
        X=np.array([[1]]),
        feature=feature,
        threshold=threshold,
        value=value,
        children_left=children_left,
        children_right=children_right,
        operator='<='
    )

    assert result[0][0][0] == 6.0
