"""
This module implements the tree inference methods
"""

import numpy as np

__all__ = ["tree_inference", "apply"]


def apply(X: np.array, tree, operator: str = "<=", random_state=None) -> np.array:
    """
    Implement the inference in a tree

    Args:
        X (np.array(float)): the array of feature vectors to infer
        tree (np.array(float)): the fitted tree
        operator (str): the splitting operator to be used

    Returns:
        np.array(int): the leaf node indices belonging to the vectors
    """

    # allocating the buffer to keep track of the inference in the tree
    node_ids = np.repeat(0, X.shape[0])
    leaf_node_flag = tree.tree_.children_left == tree.tree_.children_right

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    while not np.all(leaf_node_flag[node_ids]):
        # the indices of vectors not at leaf nodes yet
        active_indices = np.where(~leaf_node_flag[node_ids])[0]

        # the actual nodes where these vectors are
        active_nodes = node_ids[active_indices]

        # do the branching
        feature = tree.tree_.feature[active_nodes]
        threshold = tree.tree_.threshold[active_nodes]

        if operator is not None:
            if operator == "<=":
                left = X[active_indices, feature] <= threshold
            else:
                left = X[active_indices, feature] < threshold
        else:
            if random_state.randint(2) == 0:
                left = X[active_indices, feature] <= threshold
            else:
                left = X[active_indices, feature] < threshold

        # update the nodes where the vectors are
        active_nodes[left] = tree.tree_.children_left[active_nodes[left]]
        active_nodes[~left] = tree.tree_.children_right[active_nodes[~left]]

        node_ids[active_indices] = active_nodes

    return node_ids


def tree_inference(
    *, X: np.array, tree, operator: str = "<=", random_state=None
) -> np.array:
    """
    Implement the inference in a tree

    Args:
        X (np.array(float)): the array of feature vectors to infer
        tree (obj): the fitted tree
        operator (str): the splitting operator '<' or '<='

    Returns:
        np.array(int|float), np.array(int): the inferred values and leaf node ids
    """
    # allocating some buffers to keep track of the inference in the tree

    return tree.tree_.value[apply(X, tree, operator, random_state), 0, :]
