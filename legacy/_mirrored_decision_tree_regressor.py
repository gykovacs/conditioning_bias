"""
This module implements a decision tree regressor which is mirrored once fitted
"""

from ._core import mirror_tree

from sklearn.base import RegressorMixin
from sklearn.tree import DecisionTreeRegressor

__all__ =  ['MirroredDecisionTreeRegressor']

class MirroredDecisionTreeRegressor(RegressorMixin):
    """
    The mirrored decision tree regressor: mirrored once fitted
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
        self.tree = DecisionTreeRegressor(*self.args, **self.kwargs)
        self.tree.fit(X, y, sample_weight)

        self.tree = mirror_tree(self.tree)

        return self

    def predict(self, X):
        """
        Carries out the regression

        Args:
            X (np.array): the feature vectors to regress

        Returns:
            np.array: the regressed values
        """
        return self.tree.predict(-X)
