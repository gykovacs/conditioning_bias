"""
This module implements the tree inference methods
"""

import numpy as np

__all__ = ['tree_inference']

def tree_inference(
    X,
    feature,
    threshold,
    value,
    children_left,
    children_right):
    """
    Implement the inference in a tree

    Args:
        X (np.array(float)): the array of feature vectors to infer
        threshold (np.array(float)): the array of thresholds
        value (np.array(int/float)): the value vectors
        children_left (np.array(int)): the left children
        children_right (np.array(int)): the right children

    Returns:
        np.array(int/float): the inferred values
    """
    # allocating some buffers to keep track of the inference in the tree
    n_to_infer = X.shape[0]
    indices_to_infer = np.arange(n_to_infer)
    node_ids = np.repeat(0, n_to_infer)

    # allocating a buffer for the results of the inference
    predictions = np.zeros(shape=(X.shape[0], value.shape[1]))

    # allocating two indicator buffers for leaf nodes and vectors that already got to
    # leaf nodes
    leaf_nodes = np.repeat(False, children_left.shape[0])
    finished_vectors = np.repeat(False, X.shape[0])

    # initializing the leaf node indicator
    for idx, (left, right) in enumerate(zip(children_left, children_right)):
        if left == -1 and right == -1:
            leaf_nodes[idx] = True

    while n_to_infer > 0:
        # there are vectors which did not get to a leaf node

        jdx = 0
        for vector_idx in indices_to_infer[:n_to_infer]:
            # iterate through the indices of vectors to be inferred

            if leaf_nodes[node_ids[vector_idx]]:
                # if we got to a leaf node with the vector, its value is updated and the
                # finished flag is set
                predictions[vector_idx] = value[node_ids[vector_idx]]
                finished_vectors[vector_idx] = True
            else:
                # otherwise we check if the vector goes to the left hand side or right hand
                # side and update its actual position in the tree accoringly
                if X[vector_idx, feature[node_ids[vector_idx]]] <= threshold[node_ids[vector_idx]]:
                    node_ids[vector_idx] = children_left[node_ids[vector_idx]]
                else:
                    node_ids[vector_idx] = children_right[node_ids[vector_idx]]
                # the vector is recorded as one to continue the inference for
                indices_to_infer[jdx] = vector_idx
                jdx += 1

        n_to_infer = jdx

    return predictions, node_ids