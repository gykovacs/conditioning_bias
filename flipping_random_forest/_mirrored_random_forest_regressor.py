"""
This module implements a random forest regressor which is mirrored once fitted
"""

from ._core import mirror_tree

from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor

__all__ =  ['MirroredRandomForestRegressor']

class MirroredRandomForestRegressor(RegressorMixin):
    """
    The mirrored random forest regressor: mirrored once fitted
    """
    def __init__(self, **kwargs):
        """
        Constructor of the regressor

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
        self.estimator = RandomForestRegressor(**self.kwargs)
        self.estimator.fit(X, y, sample_weight)

        for idx in range(len(self.estimator.estimators_)):
            self.estimator.estimators_[idx] = mirror_tree(self.estimator.estimators_[idx])

        return self

    def predict(self, X):
        """
        Carries out the regression

        Args:
            X (np.array): the feature vectors to regress

        Returns:
            np.array: the regressed values
        """
        return self.estimator.predict(-X)
