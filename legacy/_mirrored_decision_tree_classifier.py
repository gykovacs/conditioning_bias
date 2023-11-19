"""
This module implements a decision tree classifier which is mirrored once fitted
"""

from ._core import mirror_tree

from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

__all__ = ['MirroredDecisionTreeClassifier']

class MirroredDecisionTreeClassifier(ClassifierMixin):
    """
    The mirrored decision tree classifier: mirrored once fitted
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

        self.tree = None

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
        self.tree = DecisionTreeClassifier(*self.args, **self.kwargs)
        self.tree.fit(X, y, sample_weight)
        self.tree = mirror_tree(self.tree)

        return self

    def predict_proba(self, X):
        """
        Predicts the class membership probabilities

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class membership probabilities
        """
        return self.tree.predict_proba(-X)

    def predict(self, X):
        """
        Predicts the class labels

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class labels
        """
        return self.tree.predict(-X)
