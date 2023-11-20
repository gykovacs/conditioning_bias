"""
This module implements the flipping classifiers
"""

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from ._lattice_feature import lattice_features

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


class RandomFlippingDecisionTreeClassifier:
    def __init__(self, **kwargs):
        random_state = kwargs.get('random_state')
        if not isinstance(random_state, np.random.RandomState):
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = random_state

        self.tree = RandomForestClassifier(**kwargs)

    def fit(self, X, y, lattice_features, sample_weight=None):
        self.lattice_features = lattice_features
        self.flip_mask = np.ones(len(self.lattice_features))
        self.flip_mask[self.lattice_features] = self.random_state.choice(
            [-1.0, 1.0],
            size=np.sum(self.lattice_features)
        )

        self.tree.fit(X*self.flip_mask, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.tree.predict_proba(X*self.flip_mask)

    def predict(self, X):
        return self.tree.predict(X*self.flip_mask)

class FlippingRandomForestClassifier:
    def __init__(self, mode='full', **kwargs):
        self.mode = mode

        self.n_estimators = kwargs.get('n_estimators', 100)
        self.n_half = int(self.n_estimators/2)

        self.params = kwargs | {'n_estimators': self.n_half}

        self.random_state = kwargs.get('random_state')

        if not isinstance(self.random_state, np.random.RandomState):
            self.random_state = np.random.RandomState(self.random_state)

        self.estimators_ = None

    def fit(self, X, y, sample_weight=None):
        if self.mode == 'full':
            self.forest_l = RandomForestClassifier(**self.params)
            self.forest_leq = RandomForestClassifier(**self.params)

            self.forest_l.fit(X, y, sample_weight=sample_weight)
            self.forest_leq.fit(-X, y, sample_weight=sample_weight)
        else:
            self.estimators_ = []

            self.lattice_features = lattice_features(X)

            for _ in range(self.n_estimators):
                params_extra = {'n_estimators': 1, 'random_state': self.random_state}
                estimator = RandomFlippingDecisionTreeClassifier(**(self.params | params_extra))
                self.estimators_.append(estimator.fit(X, y, lattice_features=self.lattice_features, sample_weight=sample_weight))

        return self

    def predict_proba(self, X):
        if self.mode == 'full':
            return np.mean([self.forest_l.predict_proba(X), self.forest_leq.predict_proba(-X)], axis=0)
        else:
            return np.mean([tree.predict_proba(X) for tree in self.estimators_], axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
