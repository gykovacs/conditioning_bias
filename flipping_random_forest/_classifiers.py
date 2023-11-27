"""
This module implements the classifiers with flexible operators
"""

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from ._tree_inference import tree_inference, apply
from ._lattice_feature import lattice_features

__all__ = [
    'OperatorDecisionTreeClassifier',
    'OperatorRandomForestClassifier',
    'determine_specific_operator_classification'
]

def determine_specific_operator_classification(X, y, tree, mode='normal'):
    aucs = np.array([roc_auc_score(y, X[:, idx]) for idx in range(X.shape[1])])

    if mode != 'normal':
        operators = []
        for auc in aucs:
            if auc < 0.5:
                operators.append('<=')
            else:
                operators.append('<')
        return operators


    lattice_feature_mask = lattice_features(X)

    if np.sum(lattice_feature_mask) == 0:
        return '<='

    nonzero_imp_mask = tree.feature_importances_ > 0

    imp = tree.feature_importances_[lattice_feature_mask & nonzero_imp_mask]

    if np.sum(imp) < 1e-8:
        return '<='

    imp = imp / np.sum(imp)

    weighted_aucs = imp * aucs[lattice_feature_mask & nonzero_imp_mask]

    if mode == 'normal':
        return '<' if np.sum(weighted_aucs) >= 0.5 else '<='
        #weighted_aucs = imp * aucs[lattice_feature_mask & nonzero_imp_mask]
    else:
        return '<=' if np.sum(weighted_aucs) >= 0.5 else '<'
        #weighted_aucs = (1.0 - imp) * aucs[lattice_feature_mask & nonzero_imp_mask]

    #return '<' if np.sum(weighted_aucs) >= 0.5 else '<='

class OperatorDecisionTreeClassifier:
    """
    A decision tree classifier with configurable splitting operator
    """
    def __init__(self, *, mode='<=', **kwargs):
        """
        The constructor of the classifier

        Args:
            mode (str): '<'/'<='/'avg_full'/'avg_rand'/'specific'
            kwargs (dict): the keyword arguments of the base learner decision tree
        """
        self.mode = mode
        self.tree = DecisionTreeClassifier(**kwargs)
        self.random_state = kwargs.get('random_state')
        if not isinstance(self.random_state, np.random.RandomState):
            self.random_state = np.random.RandomState(self.random_state)

    def set_mode(self, mode):
        self.mode = mode

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

        self.X_fit_ = X
        self.y_fit_ = y

        return self

    def predict_proba_operator(self, X):
        counts = tree_inference(
                X=X,
                tree=self.tree,
                operator=self.mode
            )

        return (counts.T / np.sum(counts, axis=1)).T

    def predict_proba_average(self, X):
        if self.mode == 'avg_full':
            values_le = self.tree.tree_.value[apply(X, self.tree, '<')][:, 0, :]
            values_leq = self.tree.tree_.value[apply(X, self.tree, '<=')][:, 0, :]

            values_le = (values_le.T / np.sum(values_le, axis=1)).T
            values_leq = (values_leq.T / np.sum(values_leq, axis=1)).T

            probs = np.mean(np.array([values_le, values_leq]), axis=0)

            return probs
        # avg_rand
        values = [self.tree.tree_.value[apply(X, self.tree, None, self.random_state)][:, 0, :]
                    for _ in range(10)]
        values = [(value.T / np.sum(value, axis=1)).T for value in values]

        return np.mean(np.array(values), axis=0)

    def predict_proba_specific(self, X, mode):
        if mode == 'specific':
            operator = determine_specific_operator_classification(self.X_fit_, self.y_fit_, self.tree, 'normal')
        elif mode == 'rspecific':
            operator = determine_specific_operator_classification(self.X_fit_, self.y_fit_, self.tree, 'reversed')

        counts = tree_inference(
            X=X,
            tree=self.tree,
            operator=operator
        )

        return (counts.T / np.sum(counts, axis=1)).T

    def predict_proba(self, X):
        """
        Predicting the probabilities

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the probabilities
        """

        if self.mode in ['<', '<=']:
            return self.predict_proba_operator(X)
        elif self.mode in ['avg_full', 'avg_rand']:
            return self.predict_proba_average(X)
        elif self.mode in ['specific', 'rspecific']:
            return self.predict_proba_specific(X, self.mode)

    def get_modes(self):
        return ['<', '<=', 'avg_full', 'avg_rand', 'specific']

    def predict(self, X):
        """
        Predicting the class labels

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class labels
        """

        return np.argmax(self.predict_proba(X), axis=1)

def _evaluate_trees(X, trees, operator):
    values = []
    for tree in trees:
        nodes = apply(X=X, tree=tree, operator=operator)
        values.append(tree.tree_.value[nodes][:, 0, :])

    values = [(value.T / np.sum(value, axis=1)).T for value in values]

    return np.mean(values, axis=0)

class OperatorRandomForestClassifier:
    """
    A rendom forest classifier with configurable splitting operator
    """
    def __init__(self, *, mode='<=', **kwargs):
        """
        The constructor of the classifier

        Args:
            mode (str): the operator to use ('<' or '<=')
            kwargs (dict): the keyword arguments of the base learner decision tree
        """
        self.mode = mode
        self.forest = RandomForestClassifier(**kwargs)

    def set_mode(self, mode):
        self.mode = mode

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

        self.X_fit_ = X
        self.y_fit_ = y

        return self

    def predict_proba_operator(self, X):
        counts = np.array([tree_inference(
            X=X,
            tree=tree,
            operator=self.mode
        ) for tree in self.forest.estimators_])

        probs = [(count.T / np.sum(count, axis=1)).T for count in counts]

        return np.mean(probs, axis=0)

    def predict_proba_average(self, X):
        if self.mode == 'avg_all':
            return np.mean([
                _evaluate_trees(X, self.forest.estimators_, '<'),
                _evaluate_trees(X, self.forest.estimators_, '<=')
            ], axis=0)

        n_estimators = len(self.forest.estimators_)
        n_half = int(n_estimators/2)

        return np.mean([
                _evaluate_trees(X, self.forest.estimators_[:n_half], '<'),
                _evaluate_trees(X, self.forest.estimators_[n_half:], '<=')
            ], axis=0)

    def predict_proba_specific(self, X):
        operator = determine_specific_operator_classification(self.X_fit_, self.y_fit_, self.forest)
        return _evaluate_trees(X, self.forest.estimators_, operator)

    def predict_proba(self, X):
        """
        Predicting the probabilities

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the probabilities
        """

        if self.mode in ['<', '<=']:
            return self.predict_proba_operator(X)
        elif self.mode in ['avg_all', 'avg_half']:
            return self.predict_proba_average(X)
        elif self.mode == 'specific':
            return self.predict_proba_specific(X)

    def get_modes(self):
        return ['<', '<=', 'avg_all', 'avg_half', 'specific']

    def predict(self, X):
        """
        Predicting the class labels

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class labels
        """

        return np.argmax(self.predict_proba(X), axis=1)
