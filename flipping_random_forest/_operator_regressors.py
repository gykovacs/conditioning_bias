"""
This module implements the regressors with flexible operators
"""

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from ._tree_inference import tree_inference, apply

__all__ = [
    'OperatorDecisionTreeRegressor',
    'OperatorRandomForestRegressor'
]

class OperatorDecisionTreeRegressor:
    """
    A decision tree regressor with configurable splitting operator
    """
    def __init__(self, *, operator, **kwargs):
        """
        The constructor of the regressor

        Args:
            operator (str): the operator to use ('<' or '<=')
            kwargs (dict): the keyword arguments of the base learner decision tree
        """
        self.operator = operator
        self.tree = DecisionTreeRegressor(**kwargs)

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

        return self

    def predict(self, X):
        """
        Predicting the values

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the values
        """

        return tree_inference(X=X, tree=self.tree, operator=self.operator)[:, 0]


class OperatorRandomForestRegressor:
    """
    A random forest regressor with configurable splitting operator
    """
    def __init__(self, *, operator, **kwargs):
        """
        The constructor of the regressor

        Args:
            operator (str): the operator to use ('<' or '<=')
            kwargs (dict): the keyword arguments of the base learner decision tree
        """
        self.operator = operator
        self.forest = RandomForestRegressor(**kwargs)

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

        return self

    def predict(self, X):
        """
        Predicting the values

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the values
        """

        return np.mean([tree_inference(X=X, tree=tree, operator=self.operator)
                        for tree in self.forest.estimators_], axis=0)[:, 0]
