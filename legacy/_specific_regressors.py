"""
This module implements the specific regressors
"""

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score

from ._tree_inference import apply, tree_inference
from ._lattice_feature import lattice_feature

class SpecificDecisionTreeRegressor:
    def __init__(self, **kwargs):
        self.tree = DecisionTreeRegressor(**kwargs)

    def fit(self, X, y, sample_weight=None):
        self.tree.fit(X, y, sample_weight=sample_weight)

        y_tmp = (y < np.mean(y)).astype(int)

        self.aucs = np.array([roc_auc_score(y_tmp, X[:, idx]) for idx in range(X.shape[1])])

        minority_1 = np.sum(y_tmp == 1) < np.sum(y_tmp == 0)

        self.lattice_features_ = lattice_feature(X)

        if np.sum(self.lattice_features_) == 0:
            self.operator = '<='
            return

        self.weighted_aucs = self.tree.feature_importances_[self.lattice_features_] * self.aucs[self.lattice_features_] / np.sum(self.tree.feature_importances_[self.lattice_features_])

        if minority_1:
            if np.sum(self.weighted_aucs) >= 0.5:
                self.operator = '<'
            else:
                self.operator = '<='
        else:
            if np.sum(self.weighted_aucs) >= 0.5:
                self.operator = '<='
            else:
                self.operator = '<'

        return self

    def predict(self, X):
        values = self.tree.tree_.value[apply(X, self.tree, self.operator)][:, 0, :]
        return values[:, 0]

class SpecificRandomForestRegressor:
    """
    A rendom forest classifier with configurable splitting operator
    """
    def __init__(self, **kwargs):
        """
        The constructor of the classifier

        Args:
            operator (str): the operator to use ('<' or '<=')
            kwargs (dict): the keyword arguments of the base learner decision tree
        """
        self.forest = RandomForestRegressor(**kwargs)

    def fit(self, X, y, sample_weight=None):
        """
        Fitting the classifier

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the probabilities
        """


        self.forest.fit(X, y, sample_weight)

        self.feature_importances_ = self.forest.feature_importances_

        y_tmp = (y < np.mean(y)).astype(int)

        self.aucs = np.array([roc_auc_score(y_tmp, X[:, idx]) for idx in range(X.shape[1])])

        minority_1 = np.sum(y_tmp == 1) < np.sum(y_tmp == 0)

        self.lattice_features_ = lattice_feature(X)

        if np.sum(self.lattice_features_) == 0:
            self.operator = '<='
            return

        self.weighted_aucs = self.forest.feature_importances_[self.lattice_features_] * self.aucs[self.lattice_features_] / np.sum(self.forest.feature_importances_[self.lattice_features_])

        if minority_1:
            if np.sum(self.weighted_aucs) >= 0.5:
                self.operator = '<'
            else:
                self.operator = '<='
        else:
            if np.sum(self.weighted_aucs) >= 0.5:
                self.operator = '<='
            else:
                self.operator = '<'

        return self

    def predict(self, X):
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

        return np.mean(counts, axis=0)
