"""
This module implements a flipping decision tree classifier
"""

import numpy as np

from ._core import mirror_tree

from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

__all__ = ['FlippingDecisionTreeClassifier']

class FlippingDecisionTreeClassifier(ClassifierMixin):
    """
    Decision tree classifier with attribute flipping
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor of the classifier

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
        self.tree_0 = DecisionTreeClassifier(*self.args, **self.kwargs)
        self.tree_0.fit(X, y, sample_weight)

        self.tree_1 = mirror_tree(self.tree_0)

        return self

    def predict_proba(self, X):
        """
        Predicts the class membership probabilities

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class membership probabilities
        """
        probs = np.vstack([self.tree_0.predict_proba(X)[:, 0],
                           self.tree_1.predict_proba(-X)[:, 0]]).T
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
