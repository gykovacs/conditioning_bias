"""
This module implements the averaged classifiers
"""

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from ._tree_inference import apply

class AveragedDecisionTreeClassifier:
    def __init__(self, **kwargs):
        self.tree = DecisionTreeClassifier(**kwargs)

    def fit(self, X, y, sample_weight=None):
        self.tree.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        values_le = self.tree.tree_.value[apply(X, self.tree, '<')]
        values_leq = self.tree.tree_.value[apply(X, self.tree, '<=')]

        values_le = (values_le.T / np.sum(values_le, axis=1)).T
        values_leq = (values_leq.T / np.sum(values_leq, axis=1)).T

        probs = np.mean(np.array([values_le, values_leq]), axis=0)

        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

def _evaluate_trees(X, trees, operator):
    values = []
    for tree in trees:
        nodes = apply(X=X, tree=tree, operator=operator)
        values.append(tree.tree_.value[nodes])

    values = [(value.T / np.sum(value, axis=1)).T for value in values]

    return np.mean(values, axis=0)

class AveragedRandomForestClassifier:
    def __init__(self, mode='all', **kwargs):
        self.mode = mode
        self.forest = RandomForestClassifier(**kwargs)

    def fit(self, X, y, sample_weight=None):
        self.forest.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
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

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
