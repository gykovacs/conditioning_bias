"""
This module implements the specific classifiers
"""

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from ._tree_inference import apply, tree_inference
from ._lattice_feature import lattice_feature

class SpecificDecisionTreeClassifier:
    def __init__(self, **kwargs):
        self.tree = DecisionTreeClassifier(**kwargs)

    def fit(self, X, y, sample_weight=None):
        self.tree.fit(X, y, sample_weight=sample_weight)

        self.aucs = np.array([roc_auc_score(y, X[:, idx]) for idx in range(X.shape[1])])
        #print(self.aucs, self.tree.feature_importances_)

        self.lattice_features_ = lattice_feature(X)

        if np.sum(self.lattice_features_) == 0:
            self.operator = '<='
            return

        self.weighted_aucs = self.tree.feature_importances_[self.lattice_features_] * self.aucs[self.lattice_features_] / np.sum(self.tree.feature_importances_[self.lattice_features_])

        if np.sum(self.weighted_aucs) >= 0.5:
            self.operator = '<'
        else:
            self.operator = '<='
        return self

    def predict_proba(self, X):
        values = self.tree.tree_.value[apply(X, self.tree, self.operator)][:, 0, :]

        values = (values.T / np.sum(values, axis=1)).T

        return values

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

class SpecificRandomForestClassifier:
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

        self.lattice_features_ = lattice_feature(X)

        if np.sum(self.lattice_features_) == 0:
            self.operator = '<='
            return self

        self.aucs = np.array([roc_auc_score(y, X[:, idx]) for idx in range(X.shape[1])])

        #print(self.aucs, self.tree.feature_importances_)
        self.weighted_aucs = self.forest.feature_importances_[self.lattice_features_] * self.aucs[self.lattice_features_] / np.sum(self.forest.feature_importances_[self.lattice_features_])

        if np.sum(self.weighted_aucs) >= 0.5:
            self.operator = '<'
        else:
            self.operator = '<='
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
