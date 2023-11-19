"""
This module implements the classifiers with flexible operators
"""

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from ._tree_inference import tree_inference, apply

__all__ = [
    'OperatorDecisionTreeClassifier',
    'OperatorRandomForestClassifier'
]

class OperatorDecisionTreeClassifier:
    """
    A decision tree classifier with configurable splitting operator
    """
    def __init__(self, *, operator, **kwargs):
        """
        The constructor of the classifier

        Args:
            operator (str): the operator to use ('<' or '<=')
            kwargs (dict): the keyword arguments of the base learner decision tree
        """
        self.operator = operator
        self.tree = DecisionTreeClassifier(**kwargs)

    def fit(self, X, y, sample_weight=None):
        """
        Fitting the classifier

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the probabilities
        """

        self.tree.fit(X, y, sample_weight)
        self.classes_ = self.tree.classes_
        self.feature_importances_ = self.tree.feature_importances_

        return self

    def predict_proba(self, X):
        """
        Predicting the probabilities

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the probabilities
        """

        counts = tree_inference(
            X=X,
            tree=self.tree,
            operator=self.operator
        )

        return (counts.T / np.sum(counts, axis=1)).T

    def predict(self, X):
        """
        Predicting the class labels

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class labels
        """

        return np.argmax(self.predict_proba(X), axis=1)


class OperatorRandomForestClassifier:
    """
    A rendom forest classifier with configurable splitting operator
    """
    def __init__(self, *, operator, **kwargs):
        """
        The constructor of the classifier

        Args:
            operator (str): the operator to use ('<' or '<=')
            kwargs (dict): the keyword arguments of the base learner decision tree
        """
        self.operator = operator
        self.forest = RandomForestClassifier(**kwargs)

    def fit(self, X, y, sample_weight=None):
        """
        Fitting the classifier

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the probabilities
        """

        self.forest.fit(X, y, sample_weight)
        self.classes_ = self.forest.classes_
        self.feature_importances_ = self.forest.feature_importances_

        return self

    def predict_proba(self, X):
        """
        Predicting the probabilities

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the probabilities
        """

        counts = np.array([tree_inference(
            X=X,
            tree=tree,
            operator=self.operator
        ) for tree in self.forest.estimators_])

        probs = [(count.T / np.sum(count, axis=1)).T for count in counts]

        return np.mean(probs, axis=0)

    def predict(self, X):
        """
        Predicting the class labels

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class labels
        """

        return np.argmax(self.predict_proba(X), axis=1)
