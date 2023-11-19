"""
This module implements the averaged regressors
"""

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from ._tree_inference import apply

class AveragedDecisionTreeRegressor:
    def __init__(self, **kwargs):
        self.tree = DecisionTreeRegressor(**kwargs)

    def fit(self, X, y, sample_weight=None):
        self.tree.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        values_le = self.tree.tree_.value[apply(X, self.tree, '<')][:, 0, 0]
        values_leq = self.tree.tree_.value[apply(X, self.tree, '<=')][:, 0, 0]

        probs = np.mean(np.array([values_le, values_leq]), axis=0)

        return probs

def _evaluate_trees(X, trees, operator):
    values = []
    for tree in trees:
        nodes = apply(X=X, tree=tree, operator=operator)
        values.append(tree.tree_.value[nodes][:, 0, 0])

    return np.mean(values, axis=0)

class AveragedRandomForestRegressor:
    def __init__(self, mode='all', **kwargs):
        self.mode = mode
        self.forest = RandomForestRegressor(**kwargs)

    def fit(self, X, y, sample_weight=None):
        self.forest.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self.mode == 'all':
            return np.mean([
                _evaluate_trees(X, self.forest.estimators_, '<'),
                _evaluate_trees(X, self.forest.estimators_, '<=')
            ], axis=0)

        n_estimators = len(self.forest.estimators_)
        n_half = int(n_estimators/2)

        return np.mean([
                _evaluate_trees(X, self.forest.estimators_[:n_half], '<'),
                _evaluate_trees(X, self.forest.estimators_[n_half:], '<=')
            ], axis=0)
