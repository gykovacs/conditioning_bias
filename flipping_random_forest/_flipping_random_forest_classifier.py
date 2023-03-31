"""
This module implements a flipping random forest classifier
"""

import copy

import numpy as np

from ._core import mirror_tree

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

__all__ = ['FlippingRandomForestClassifier']

class FlippingRandomForestClassifier(ClassifierMixin):
    """
        The constructor of the flipping forest classifier
        
        Args:
            kwargs (dict): the arguments of the classifier
        """
    def __init__(self, **kwargs):
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
        self.positive = RandomForestClassifier(**self.kwargs_pos)
        self.negative = RandomForestClassifier(**self.kwargs_neg)

        self.positive.fit(X, y, sample_weight)
        self.negative.fit(-X, y, sample_weight)

        return self

    def predict_proba(self, X):
        """
        Predicts the class membership probabilities

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class membership probabilities
        """
        probs = np.vstack([self.positive.predict_proba(X)[:, 0],
                           self.negative.predict_proba(-X)[:, 0]]).T
        probs = np.mean(probs, axis=1)

        return np.vstack([probs, 1.0 - probs]).T

    def predict(self, X):
        """
        Predicts the class labels

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class labels
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5)*1
