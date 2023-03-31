"""
This module implements a flipping random forest regressor
"""

import copy

import numpy as np

from ._core import mirror_tree

from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor

__all__ =  ['FlippingRandomForestRegressor']

class FlippingRandomForestRegressor(RegressorMixin):
    """
    The flipping forest regressor
    """
    def __init__(self, **kwargs):
        """
        The constructor of the flipping forest regressor
        
        Args:
            kwargs (dict): the arguments of the regressor
        """
        if 'n_estimators' in kwargs:
            n_estimators = kwargs['n_estimators']
        else:
            n_estimators = 100

        n_half_estimators = int(np.round(n_estimators/2))

        self.kwargs_pos = copy.deepcopy(kwargs)
        self.kwargs_pos['n_estimators'] = n_half_estimators
        self.kwargs_neg = copy.deepcopy(self.kwargs_pos)
        if 'random_state' in self.kwargs_pos:
            self.kwargs_neg['random_state'] = self.kwargs_pos['random_state'] + 1

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
        self.positive = RandomForestRegressor(**self.kwargs_pos)
        self.negative = RandomForestRegressor(**self.kwargs_neg)

        self.positive.fit(X, y, sample_weight)
        self.negative.fit(-X, y, sample_weight)

        return self

    def predict(self, X):
        """
        Carries out the regression

        Args:
            X (np.array): the feature vectors to regress

        Returns:
            np.array: the regressed values
        """
        probs = np.vstack([self.positive.predict(X),
                           self.negative.predict(-X)]).T
        probs = np.mean(probs, axis=1)

        return probs
