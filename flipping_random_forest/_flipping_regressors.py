"""
This module implements the flipping regressors
"""

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

class FlippingDecisionTreeRegressor:
    def __init__(self, **kwargs):
        self.tree_l = DecisionTreeRegressor(**kwargs)
        self.tree_leq = DecisionTreeRegressor(**kwargs)

    def fit(self, X, y, sample_weight=None):
        self.tree_l.fit(X, y, sample_weight=sample_weight)
        self.tree_leq.fit(-X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return np.mean([self.tree_l.predict(X), self.tree_leq.predict(-X)], axis=0)

class FlippingRandomForestRegressor:
    def __init__(self, mode='full', **kwargs):
        n_estimators = kwargs.get('n_estimators', 100)
        n_half = int(n_estimators/2)
        params = kwargs | {'n_estimators': n_half}
        self.forest_l = RandomForestRegressor(**params)
        self.forest_leq = RandomForestRegressor(**params)

    def fit(self, X, y, sample_weight=None):
        self.forest_l.fit(X, y, sample_weight=sample_weight)
        self.forest_leq.fit(-X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return np.mean([self.forest_l.predict(X), self.forest_leq.predict(-X)], axis=0)
