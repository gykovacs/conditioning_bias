"""
Testing the tree inference methods
"""

import numpy as np

from conditioning_bias import tree_inference

def test_tree_inference():
    """
    Testing the tree inference
    """

    class TreeDataMock:
        def __init__(self, feature, threshold, value, children_left, children_right):
            self.feature = feature
            self.threshold = threshold
            self.value = value
            self.children_left = children_left
            self.children_right = children_right

    class TreeMock:
        def __init__(self, treedata):
            self.tree_ = treedata

    feature = np.array([0, -1, -1])
    threshold = np.array([1, -1, -1])
    value = np.array([[[5]], [[6]], [[3]]])
    children_left = np.array([1, -1, -1])
    children_right = np.array([2, -1, -1])

    tree = TreeMock(
        TreeDataMock(
            feature=feature,
            threshold=threshold,
            value=value,
            children_left=children_left,
            children_right=children_right
        )
    )

    result = tree_inference(
        X=np.array([[1]]),
        tree=tree,
        operator='<'
    )

    assert result[0][0] == 3.0

    result = tree_inference(
        X=np.array([[1]]),
        tree=tree,
        operator='<='
    )

    assert result[0][0] == 6.0

    result = tree_inference(
        X=np.array([[1]]),
        tree=tree,
        operator=None,
        random_state=4
    )

    assert result[0][0] == 6.0

    result = tree_inference(
        X=np.array([[1]]),
        tree=tree,
        operator=None,
        random_state=5
    )

    assert result[0][0] == 3.0
