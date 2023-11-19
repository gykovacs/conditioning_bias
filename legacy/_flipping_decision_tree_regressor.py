"""
This module implements a flipping decision tree regressor
"""

import numpy as np

from ._core import mirror_tree

from sklearn.base import RegressorMixin
from sklearn.tree import DecisionTreeRegressor

__all__ =  ['FlippingDecisionTreeRegressor']

class FlippingDecisionTreeRegressor(RegressorMixin):
    """
    Decision tree regressor with attribute flipping
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor of the regressor

        Args:
            args: positional arguments passed to the underlying decision trees
            kwargs: keyword arguments passed to the underlying decision trees
        """
        self.args = args
        self.kwargs = kwargs

        self.tree_0 = None
        self.tree_1 = None

    def fit(self, X, y, sample_weight=None):
        """
        Fits the predictor

        Args:
            X (np.array): the feature vectors
            y (np.array): the target labels
            sample_weight (None/np.array): the sample weights

        Returns:
            self: the fitted object
        """
        self.tree_0 = DecisionTreeRegressor(*self.args, **self.kwargs)
        self.tree_0.fit(X, y, sample_weight)

        self.tree_1 = mirror_tree(self.tree_0)

        return self

    def predict(self, X):
        """
        Carries out the regression

        Args:
            X (np.array): the feature vectors to regress

        Returns:
            np.array: the regressed values
        """
        probs = np.vstack([
            self.tree_0.predict(X),
            self.tree_1.predict(-X)
        ]).T
        probs = np.mean(probs, axis=1)

        return probs
