"""
This module implements the tree inference methods
"""

import numpy as np

__all__ = ["tree_inference", "apply"]


def apply(X: np.array, tree, operator: str = "<=", random_state=None, max_depth=None) -> np.array:
    """
    Implement the inference in a tree

    Args:
        X (np.array(float)): the array of feature vectors to infer
        tree (np.array(float)): the fitted tree
        operator (str): the splitting operator to be used
        max_depth (int|None): if set, traversal stops at this depth (root=0) even if the node
                            is not an actual leaf, and the node's own recorded statistics are
                            used as if it were one. NOTE: growing a tree once with no depth limit
                            and truncating traversal at a given depth is only exactly equivalent
                            to a real ``max_depth=max_depth`` refit when split selection never
                            needs to break a tie; in practice, sklearn's tree builder uses the
                            shared random_state to break ties between equally-good candidate
                            splits, and its depth-first construction order consumes that RNG
                            stream differently depending on how deep the tree is allowed to grow
                            elsewhere, so refits and truncated unconstrained trees can diverge
                            structurally whenever a tie occurs (validated empirically -- see
                            study/revision/validate_depth_truncation.py). Useful for inspecting an
                            already-fitted tree at a shallower depth, not as a substitute for
                            refitting with a smaller max_depth.

    Returns:
        np.array(int): the leaf (or truncation) node indices belonging to the vectors
    """

    # allocating the buffer to keep track of the inference in the tree
    node_ids = np.repeat(0, X.shape[0])
    depths = np.zeros(X.shape[0], dtype=int)
    leaf_node_flag = tree.tree_.children_left == tree.tree_.children_right

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    def _not_done():
        stopped = leaf_node_flag[node_ids]
        if max_depth is not None:
            stopped = stopped | (depths >= max_depth)
        return ~np.all(stopped)

    while _not_done():
        # the indices of vectors not at leaf nodes (or the depth limit) yet
        stopped = leaf_node_flag[node_ids]
        if max_depth is not None:
            stopped = stopped | (depths >= max_depth)
        active_indices = np.where(~stopped)[0]

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
        depths[active_indices] += 1

    return node_ids


def tree_inference(
    *, X: np.array, tree, operator: str = "<=", random_state=None, max_depth=None
) -> np.array:
    """
    Implement the inference in a tree

    Args:
        X (np.array(float)): the array of feature vectors to infer
        tree (obj): the fitted tree
        operator (str): the splitting operator '<' or '<='
        max_depth (int|None): optional traversal depth limit, see ``apply``

    Returns:
        np.array(int|float), np.array(int): the inferred values and leaf node ids
    """
    # allocating some buffers to keep track of the inference in the tree

    return tree.tree_.value[apply(X, tree, operator, random_state, max_depth), 0, :]
