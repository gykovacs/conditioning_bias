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
    'OperatorRandomForestClassifier'
]

class OperatorDecisionTreeClassifier:
    """
    A decision tree classifier with configurable splitting operator
    """
    def __init__(self, *, mode: str = '<=', **kwargs):
        """
        The constructor of the classifier

        Args:
            mode (str): '<'/'<='/'avg_full'/'avg_rand'
            kwargs (dict): the keyword arguments of the base learner decision tree
        """
        self.mode = mode
        self.tree = DecisionTreeClassifier(**kwargs)
        self.random_state = kwargs.get('random_state')
        if not isinstance(self.random_state, np.random.RandomState):
            self.random_state = np.random.RandomState(self.random_state)

    def set_mode(self, mode: str):
        """
        Set the mode of operation

        Args:
            mode (str): '<'/'<='/'avg_full'/'avg_rand'

        Returns:
            OperatorDecisionTreeClassifier: the modified object
        """
        self.mode = mode
        return self

    def fit(self, X: np.array, y: np.array, sample_weight: np.array = None):
        """
        Fitting the classifier

        Args:
            X (np.array): the feature vectors to predict
            y (np.array): the target label
            sample_weight (np.array|None): the sample weights to be used

        Returns:
            OperatorDecisionTreeClassifier: the fitted object
        """

        self.tree.fit(X, y, sample_weight)

        self.classes_ = self.tree.classes_
        self.feature_importances_ = self.tree.feature_importances_

        self.X_fit_ = X
        self.y_fit_ = y

        return self

    def predict_proba_operator(self, X: np.array):
        """
        Predict probabilities with a specific operator

        Args:
            X (np.array): the feature vectors to be predicted

        Returns:
            np.array: the predicted probabilities
        """
        counts = tree_inference(
                X=X,
                tree=self.tree,
                operator=self.mode
            )

        return (counts.T / np.sum(counts, axis=1)).T

    def predict_proba_average(self, X):
        """
        Predict probabilities with averaging

        Args:
            X (np.array): the feature vectors to be predicted

        Returns:
            np.array: the predicted probabilities
        """
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

    def get_modes(self):
        """
        Return the list of operating modes

        Returns:
            list(str): the list of supported operating modes
        """
        return ['<', '<=', 'avg_full', 'avg_rand']

    def predict(self, X):
        """
        Predicting the class labels

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class labels
        """

        return np.argmax(self.predict_proba(X), axis=1)

def _evaluate_trees(X: np.array, trees: list, operator: str):
    """
    Evaluates a list if trees

    Args:
        X (np.array): the feature vectors to predict
        trees (list(str)): the list of trees to use for prediction
        operator (str): the operator to be used during the prediction

    Returns:
        np.array: the predicted probabilities
    """
    values = []
    for tree in trees:
        nodes = apply(X=X, tree=tree, operator=operator)
        values.append(tree.tree_.value[nodes][:, 0, :])

    values = [(value.T / np.sum(value, axis=1)).T for value in values]

    return np.mean(values, axis=0)

class OperatorRandomForestClassifier:
    """
    A random forest classifier with configurable splitting operator
    """
    def __init__(self, *, mode: str = '<=', **kwargs):
        """
        The constructor of the classifier

        Args:
            mode (str): the operator to use ('<'/'<='/'avg_all'/'avg_half')
            kwargs (dict): the keyword arguments of the base learner decision tree
        """
        self.mode = mode
        self.forest = RandomForestClassifier(**kwargs)

    def set_mode(self, mode: str):
        """
        Set the mode of operation

        Args:
            mode (str): '<'/'<='/'avg_full'/'avg_rand'

        Returns:
            OperatorRandomForestClassifier: the modified object
        """
        self.mode = mode
        return self

    def fit(self, X: np.array, y: np.array, sample_weight: np.array = None):
        """
        Fitting the classifier

        Args:
            X (np.array): the feature vectors to predict
            y (np.array): the target label
            sample_weight (np.array|None): the sample weights to be used

        Returns:
            OperatorRandomForestClassifier: the fitted object
        """

        self.forest.fit(X, y, sample_weight)
        self.classes_ = self.forest.classes_
        self.feature_importances_ = self.forest.feature_importances_

        self.X_fit_ = X
        self.y_fit_ = y

        return self

    def predict_proba_operator(self, X: np.array):
        """
        Predict probabilities with a specific operator

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the predicted probabilities
        """
        counts = np.array([tree_inference(
            X=X,
            tree=tree,
            operator=self.mode
        ) for tree in self.forest.estimators_])

        probs = [(count.T / np.sum(count, axis=1)).T for count in counts]

        return np.mean(probs, axis=0)

    def predict_proba_average(self, X: np.array):
        """
        Predict probabilities by averaging

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the predicted probabilities
        """
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

    def predict_proba(self, X: np.array):
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

    def get_modes(self):
        """
        Return the list of operating modes

        Returns:
            list(str): the list of operating modes
        """
        return ['<', '<=', 'avg_all', 'avg_half']

    def predict(self, X: np.array):
        """
        Predicting the class labels

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class labels
        """

        return np.argmax(self.predict_proba(X), axis=1)
