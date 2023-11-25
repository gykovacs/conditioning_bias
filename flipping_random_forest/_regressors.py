"""
This module implements the regressors with flexible operators
"""

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

from ._tree_inference import tree_inference, apply
from ._lattice_feature import lattice_features

__all__ = [
    'OperatorDecisionTreeRegressor',
    'OperatorRandomForestRegressor',
    'determine_specific_operator_regression'
]

def determine_specific_operator_regression(X, y, tree):
    y_tmp = (y < np.mean(y)).astype(int)

    aucs = np.array([roc_auc_score(y_tmp, X[:, idx]) for idx in range(X.shape[1])])

    minority_1 = np.sum(y_tmp == 1) < np.sum(y_tmp == 0)

    lattice_feature_mask = lattice_features(X)

    if np.sum(lattice_feature_mask) == 0:
        return '<='

    imp = tree.feature_importances_[lattice_feature_mask]
    weighted_aucs = imp * aucs[lattice_feature_mask] / np.sum(imp)

    if minority_1:
        return '<' if np.sum(weighted_aucs) >= 0.5 else '<='
    else:
        return '<=' if np.sum(weighted_aucs) >= 0.5 else '<'

class OperatorDecisionTreeRegressor:
    """
    A decision tree regressor with configurable splitting operator
    """
    def __init__(self, *, mode='<=', **kwargs):
        """
        The constructor of the regressor

        Args:
            mode (str): the operator to use ('<' or '<=')
            kwargs (dict): the keyword arguments of the base learner decision tree
        """
        self.mode = mode
        self.tree = DecisionTreeRegressor(**kwargs)
        self.random_state = kwargs.get('random_state')
        if not isinstance(self.random_state, np.random.RandomState):
            self.random_state = np.random.RandomState(self.random_state)

    def set_mode(self, mode):
        self.mode = mode

    def fit(self, X, y, sample_weight=None):
        """
        Fitting the regressor

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the probabilities
        """

        self.tree.fit(X, y, sample_weight)
        self.feature_importances_ = self.tree.feature_importances_

        self.X_fit_ = X
        self.y_fit_ = y

        return self

    def predict_operator(self, X):
        return tree_inference(
                X=X,
                tree=self.tree,
                operator=self.mode
            )[:, 0]

    def predict_average(self, X):
        if self.mode == 'avg_full':
            values_le = self.tree.tree_.value[apply(X, self.tree, '<')][:, 0, 0]
            values_leq = self.tree.tree_.value[apply(X, self.tree, '<=')][:, 0, 0]

            values = np.mean(np.array([values_le, values_leq]), axis=0)

            return values
        else:
            values = [self.tree.tree_.value[apply(X, self.tree, None, self.random_state)][:, 0, 0]
                        for _ in range(10)]
            values = np.mean(values, axis=0)

            return values

    def predict_specific(self, X):
        operator = determine_specific_operator_regression(self.X_fit_, self.y_fit_, self.tree)

        return tree_inference(
                X=X,
                tree=self.tree,
                operator=operator
            )[:, 0]


    def predict(self, X):
        """
        Predicting the values

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the values
        """

        if self.mode in ['<', '<=']:
            return self.predict_operator(X)
        elif self.mode in ['avg_full', 'avg_half']:
            return self.predict_average(X)
        elif self.mode == 'specific':
            return self.predict_specific(X)

    def get_modes(self):
        return ['<', '<=', 'avg_full', 'avg_half', 'specific']

def _evaluate_trees(X, trees, operator):
    values = []
    for tree in trees:
        nodes = apply(X=X, tree=tree, operator=operator)
        values.append(tree.tree_.value[nodes][:, 0, 0])

    return np.mean(values, axis=0)

class OperatorRandomForestRegressor:
    """
    A random forest regressor with configurable splitting operator
    """
    def __init__(self, *, mode='<=', **kwargs):
        """
        The constructor of the regressor

        Args:
            mode (str): the operator to use ('<' or '<=')
            kwargs (dict): the keyword arguments of the base learner decision tree
        """
        self.mode = mode
        self.forest = RandomForestRegressor(**kwargs)

    def set_mode(self, mode):
        self.mode = mode

    def fit(self, X, y, sample_weight=None):
        """
        Fitting the regressor

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the probabilities
        """

        self.forest.fit(X, y, sample_weight)
        self.feature_importances_ = self.forest.feature_importances_

        self.X_fit_ = X
        self.y_fit_ = y

        return self

    def predict_operator(self, X):
        return np.mean([tree_inference(X=X, tree=tree, operator=self.mode)
                        for tree in self.forest.estimators_], axis=0)[:, 0]

    def predict_average(self, X):
        if self.mode == 'avg_all':
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

    def predict_specific(self, X):
        operator = determine_specific_operator_regression(self.X_fit_, self.y_fit_, self.forest)
        return np.mean([tree_inference(X=X, tree=tree, operator=operator)
                        for tree in self.forest.estimators_], axis=0)[:, 0]

    def predict(self, X):
        """
        Predicting the values

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the values
        """

        if self.mode in ['<', '<=']:
            return self.predict_operator(X)
        elif self.mode in ['avg_all', 'avg_half']:
            return self.predict_average(X)
        elif self.mode == 'specific':
            return self.predict_specific(X)

    def get_modes(self):
        return ['<', '<=', 'avg_all', 'avg_half', 'specific']
