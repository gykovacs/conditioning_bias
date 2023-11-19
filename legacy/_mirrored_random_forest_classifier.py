"""
This module implements a random forest classifier which is mirrored once fitted
"""

from ._core import mirror_tree

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

__all__ = ['MirroredRandomForestClassifier']

class MirroredRandomForestClassifier(ClassifierMixin):
    """
    The mirrored random forest classifier: mirrored once fitted
    """
    def __init__(self, **kwargs):
        """
        Constructor of the classifier

        Args:
            kwargs: keyword arguments passed to the underlying decision trees
        """
        self.kwargs = kwargs

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
        self.estimator = RandomForestClassifier(**self.kwargs)
        self.estimator.fit(X, y, sample_weight)

        for idx in range(len(self.estimator.estimators_)):
            self.estimator.estimators_[idx] = mirror_tree(self.estimator.estimators_[idx])

        return self

    def predict_proba(self, X):
        """
        Predicts the class membership probabilities

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class membership probabilities
        """
        return self.estimator.predict_proba(-X)

    def predict(self, X):
        """
        Predicts the class labels

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class labels
        """
        return self.estimator.predict(-X)
