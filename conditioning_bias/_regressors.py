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
    'OperatorRandomForestRegressor'
]

class OperatorDecisionTreeRegressor:
    """
    A decision tree regressor with configurable splitting operator
    """
    def __init__(self, *, mode='<=', **kwargs):
        """
        The constructor of the regressor

        Args:
            mode (str): the operator to use ('<'/'<='/'avg_full'/'avg_rand')
            kwargs (dict): the keyword arguments of the base learner decision tree
        """
        self.mode = mode
        self.tree = DecisionTreeRegressor(**kwargs)
        self.random_state = kwargs.get('random_state')
        if not isinstance(self.random_state, np.random.RandomState):
            self.random_state = np.random.RandomState(self.random_state)

    def set_mode(self, mode: str):
        """
        Set the mode of operation

        Args:
            mode (str): the mode of operation ('<'/'<='/'avg_full'/'avg_rand')

        Returns:
            obj: the adjusted object
        """
        self.mode = mode
        return self

    def fit(self, X: np.array, y: np.array, sample_weight: np.array = None):
        """
        Fitting the classifier

        Args:
            X (np.array): the feature vectors to predict
            y (np.array): the target label
            sample_weight (np.array|None): the sample weights to be used

        Returns:
            OperatorDecisionTreeRegressor: the fitted object
        """

        self.tree.fit(X, y, sample_weight)
        self.feature_importances_ = self.tree.feature_importances_

        self.X_fit_ = X
        self.y_fit_ = y

        return self

    def predict_operator(self, X: np.array):
        """
        Predicts with a specific operator

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the predictions
        """
        return tree_inference(
                X=X,
                tree=self.tree,
                operator=self.mode
            )[:, 0]

    def predict_average(self, X: np.array):
        """
        Predicts by averaging

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the predictions
        """
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

    def predict(self, X: np.array):
        """
        Predicting the values

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the values
        """

        if self.mode in ['<', '<=']:
            return self.predict_operator(X)
        elif self.mode in ['avg_full', 'avg_rand']:
            return self.predict_average(X)

    def get_modes(self):
        return ['<', '<=', 'avg_full', 'avg_rand']

def _evaluate_trees(X: np.array, trees: list, operator: str):
    """
    Evaluates a list if trees

    Args:
        X (np.array): the feature vectors to predict
        trees (list(str)): the list of trees to use for prediction
        operator (str): the operator to be used during the prediction

    Returns:
        np.array: the predictions
    """
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
            mode (str): the operator to use ('<'/'<='/'avg_full'/'avg_half')
            kwargs (dict): the keyword arguments of the base learner decision tree
        """
        self.mode = mode
        self.forest = RandomForestRegressor(**kwargs)

    def set_mode(self, mode):
        """
        Set the mode of operation

        Args:
            mode (str): '<'/'<='/'avg_full'/'avg_half'

        Returns:
            OperatorRandomForestRegressor: the modified object
        """
        self.mode = mode
        return self

    def fit(self, X, y, sample_weight=None):
        """
        Fitting the regressor

        Args:
            X (np.array): the feature vectors to predict
            y (np.array): the target label
            sample_weight (np.array|None): the sample weights to be used

        Returns:
            OperatorRandomForestRegressor: the fitted object
        """

        self.forest.fit(X, y, sample_weight)
        self.feature_importances_ = self.forest.feature_importances_

        self.X_fit_ = X
        self.y_fit_ = y

        return self

    def predict_operator(self, X):
        """
        Predicts with a specific operator

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the predictions
        """
        return np.mean([tree_inference(X=X, tree=tree, operator=self.mode)
                        for tree in self.forest.estimators_], axis=0)[:, 0]

    def predict_average(self, X):
        """
        Predicts by averaging

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the predictions
        """
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

    def get_modes(self):
        """
        Return the list of operating modes

        Returns:
            list(str): the list of operating modes
        """
        return ['<', '<=', 'avg_all', 'avg_half']
