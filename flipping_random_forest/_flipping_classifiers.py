"""
This module implements the flipping classifiers
"""

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class FlippingDecisionTreeClassifier:
    def __init__(self, **kwargs):
        self.tree_l = DecisionTreeClassifier(**kwargs)
        self.tree_leq = DecisionTreeClassifier(**kwargs)

    def fit(self, X, y, sample_weight=None):
        self.tree_l.fit(X, y, sample_weight=sample_weight)
        self.tree_leq.fit(-X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return np.mean([self.tree_l.predict_proba(X), self.tree_leq.predict_proba(-X)], axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

class FlippingRandomForestClassifier:
    def __init__(self, mode='full', **kwargs):
        n_estimators = kwargs.get('n_estimators', 100)
        n_half = int(n_estimators/2)
        params = kwargs | {'n_estimators': n_half}
        self.forest_l = RandomForestClassifier(**params)
        self.forest_leq = RandomForestClassifier(**params)

    def fit(self, X, y, sample_weight=None):
        self.forest_l.fit(X, y, sample_weight=sample_weight)
        self.forest_leq.fit(-X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return np.mean([self.forest_l.predict_proba(X), self.forest_leq.predict_proba(-X)], axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
