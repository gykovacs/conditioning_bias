"""
This module implements the flipping random forest classifier and regressor
"""

import copy

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.tree import (DecisionTreeClassifier,
                            DecisionTreeRegressor)
from sklearn.ensemble import (RandomForestClassifier,
                                RandomForestRegressor)

__all__ = ['MirroredDecisionTreeClassifier',
            'MirroredDecisionTreeRegressor',
            'FlippingDecisionTreeClassifier',
            'FlippingDecisionTreeRegressor',
            'FlippingRandomForestClassifier',
            'FlippingRandomForestRegressor',
            'MirroredRandomForestRegressor',
            'MirroredRandomForestClassifier',
            'RandomStateMixin',
            'fit_forest',
            'FlippingBaggingBase',
            'mirror_tree',
            'dummy_data']

def dummy_data():
    data = pd.DataFrame({
        'criminal_record': [0,  0,  1,  1],
        'income':          [40, 60, 40, 80],
        'years_in_job':    [4,  6,  6,  2],
        'defaulted':       [1,  0,  0,  0]
    })

    dataset = {}

    dataset['data'] = data[['criminal_record', 'income', 'years_in_job']].values
    dataset['target'] = data['defaulted'].values

    return dataset

def fit_forest(params, X, y, class_, sample_weight=None):
    """
    Fitting one tree of the forest

    Args:
        params (dict): the parameters of the tree
        X (np.array): the independent feature vectors
        y (np.array): the dependent target values
        class_ (class): the tree class to fit
        sample_weight (None/np.array): the sample weights
    """
    estimator = class_(**params)
    return estimator.fit(X, y, sample_weight=sample_weight)

class RandomStateMixin:
    """
    Mixin to set random state
    """
    def __init__(self, random_state):
        """
        Constructor of the mixin

        Args:
            random_state (int/np.random.RandomState/None): the random state
                                                                initializer
        """
        self.set_random_state(random_state)

    def set_random_state(self, random_state):
        """
        sets the random_state member of the object

        Args:
            random_state (int/np.random.RandomState/None): the random state
                                                                initializer
        """

        self._random_state_init = random_state

        if random_state is None:
            self.random_state = np.random
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        elif random_state is np.random:
            self.random_state = random_state
        else:
            raise ValueError(
                "random state cannot be initialized by " + str(random_state))

    def get_params(self, deep=False):
        """
        Returns the parameters of the object.

        Args:
            deep (bool): deep parameters

        Returns:
            dict: the parameter dictionary
        """
        _ = deep # disabling pylint reporting
        return {'random_state': self._random_state_init}

def mirror_tree(tree):
    """
    Mirrors an sklearn tree

    Args:
        tree (obj): the tree to mirror

    Returns:
        obj: the mirrored tree
    """
    new_tree = copy.deepcopy(tree)

    for idx in range(len(new_tree.tree_.children_left)):
        left = new_tree.tree_.children_left[idx]
        right = new_tree.tree_.children_right[idx]

        new_tree.tree_.children_left[idx] = right
        new_tree.tree_.children_right[idx] = left

        new_tree.tree_.threshold[idx] = (-1) * new_tree.tree_.threshold[idx]

    return new_tree

class MirroredDecisionTreeClassifier(ClassifierMixin):
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

class FlippingDecisionTreeClassifier(ClassifierMixin):
    """
    Decision tree classifier with attribute flipping
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

        self.tree_0 = None
        self.tree_1 = None

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
        self.tree_0 = DecisionTreeClassifier(*self.args, **self.kwargs)
        self.tree_0.fit(X, y, sample_weight)

        self.tree_1 = mirror_tree(self.tree_0)

        return self

    def predict_proba(self, X):
        """
        Predicts the class membership probabilities

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class membership probabilities
        """
        probs = np.vstack([
            self.tree_0.predict_proba(X)[:, 0],
            self.tree_1.predict_proba(-X)[:, 0]
        ]).T
        probs = np.mean(probs, axis=1)

        return np.vstack([probs, 1.0 - probs]).T

    def predict(self, X):
        """
        Predicts the class labels

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class labels
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5)*1

class MirroredDecisionTreeRegressor(RegressorMixin):
    """
    Decision tree regressor with attribute flipping
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

class FlippingDecisionTreeRegressor(RegressorMixin):
    """
    Decision tree regressor with attribute flipping
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

        self.tree_0 = None
        self.tree_1 = None

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
        self.tree_0 = DecisionTreeRegressor(*self.args, **self.kwargs)
        self.tree_0.fit(X, y, sample_weight)

        self.tree_1 = mirror_tree(self.tree_0)

        return self

    def predict(self, X):
        """
        Carries out the regression

        Args:
            X (np.array): the feature vectors to regress

        Returns:
            np.array: the regressed values
        """
        probs = np.vstack([
            self.tree_0.predict(X),
            self.tree_1.predict(-X)
        ]).T
        probs = np.mean(probs, axis=1)

        return probs

class MirroredRandomForestRegressor(RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y, sample_weight=None):
        self.estimator = RandomForestRegressor(**self.kwargs)
        self.estimator.fit(X, y, sample_weight)

        for idx in range(len(self.estimator.estimators_)):
            self.estimator.estimators_[idx] = mirror_tree(self.estimator.estimators_[idx])

        return self

    def predict(self, X):
        return self.estimator.predict(-X)

class MirroredRandomForestClassifier(ClassifierMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y, sample_weight=None):
        self.estimator = RandomForestClassifier(**self.kwargs)
        self.estimator.fit(X, y, sample_weight)

        for idx in range(len(self.estimator.estimators_)):
            self.estimator.estimators_[idx] = mirror_tree(self.estimator.estimators_[idx])

        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(-X)

    def predict(self, X):
        return self.estimator.predict(-X)

class FlippingRandomForestRegressor(RegressorMixin):
    def __init__(self, **kwargs):
        if 'n_estimators' in kwargs:
            n_estimators = kwargs['n_estimators']
        else:
            n_estimators = 100

        n_half_estimators = int(np.round(n_estimators/2))

        self.kwargs_tmp = copy.deepcopy(kwargs)
        self.kwargs_tmp['n_estimators'] = n_half_estimators

        if 'flipping' in self.kwargs_tmp:
            del self.kwargs_tmp['flipping']

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
        self.positive = RandomForestRegressor(**self.kwargs_tmp)
        self.negative = RandomForestRegressor(**self.kwargs_tmp)

        self.positive.fit(X, y, sample_weight)
        self.negative.fit(-X, y, sample_weight)

        return self

    def predict(self, X):
        """
        Carries out the regression

        Args:
            X (np.array): the feature vectors to regress

        Returns:
            np.array: the regressed values
        """
        probs = np.vstack([
            self.positive.predict(X),
            self.negative.predict(-X)
        ]).T
        probs = np.mean(probs, axis=1)

        return probs

class FlippingRandomForestClassifier(ClassifierMixin):
    def __init__(self, **kwargs):
        if 'n_estimators' in kwargs:
            n_estimators = kwargs['n_estimators']
        else:
            n_estimators = 100

        n_half_estimators = int(np.round(n_estimators/2))

        self.kwargs_tmp = copy.deepcopy(kwargs)
        self.kwargs_tmp['n_estimators'] = n_half_estimators

        if 'flipping' in self.kwargs_tmp:
            del self.kwargs_tmp['flipping']

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
        self.positive = RandomForestClassifier(**self.kwargs_tmp)
        self.negative = RandomForestClassifier(**self.kwargs_tmp)

        self.positive.fit(X, y, sample_weight)
        self.negative.fit(-X, y, sample_weight)

        return self

    def predict_proba(self, X):
        """
        Predicts the class membership probabilities

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class membership probabilities
        """
        probs = np.vstack([
            self.positive.predict_proba(X)[:, 0],
            self.negative.predict_proba(-X)[:, 0]
        ]).T
        probs = np.mean(probs, axis=1)

        return np.vstack([probs, 1.0 - probs]).T

    def predict(self, X):
        """
        Predicts the class labels

        Args:
            X (np.array): the feature vectors to predict

        Returns:
            np.array: the class labels
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5)*1

class FlippingBaggingBase(RandomStateMixin):
    """
    Base class for flipping forests
    """
    def __init__(self,
                    *,
                    n_estimators=100,
                    bootstrap=True,
                    bootstrap_features=False,
                    min_samples_leaf=1,
                    max_depth=None,
                    max_features='sqrt',
                    splitter='best',
                    n_jobs=1,
                    flipping=None,
                    random_state=None):
        """
        Constructor of the base class

        Args:
            n_estimators (int): the number of estimators
            bootstrap (bool): whether to do bootstrap sampling
            bootstrap_features (bool): whether to bootstrap the features
            min_samples_leaf (int): the minimum number of samples on leaf nodes
            max_depth (int): the maximum depth
            max_features (int/str): the number of features to test in nodes ('sqrt'/'auto')
            splitter (str): the splitter to be used ('random'/'best')
            n_jobs (int): the number of training jobs
            flipping (str): None/'coordinate'/'full' - the mode of flipping
            random_state (None/int/np.random.RandomState): the random state to be used
        """
        RandomStateMixin.__init__(self, random_state)

        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.splitter = splitter
        self.flipping = flipping
        self.n_jobs = n_jobs

        self.flippings_ = []

    def get_params(self, deep=False):
        """
        Returns the parameters of the object

        Returns:
            dict: the parameters of the object
        """
        _ = deep
        return {'n_estimators': self.n_estimators,
                'bootstrap': self.bootstrap,
                'bootstrap_features': self.bootstrap_features,
                'min_samples_leaf': self.min_samples_leaf,
                'max_depth': self.max_depth,
                'max_features': self.max_features,
                'splitter': self.splitter,
                'flipping': self.flipping,
                'n_jobs': self.n_jobs,
                **RandomStateMixin.get_params(self)}

    def determine_max_features(self, n_features):
        """
        Determines the maximum number of features to be used

        Args:
            n_features (str/int): the specification of the number of features

        Returns:
            int: the number of features to be tested at the nodes
        """
        max_features = 'sqrt'

        if self.max_features == 'sqrt':
            pass
        elif self.max_features == 'all':
            max_features = n_features
        elif self.max_features == 'sqrt-1':
            max_features = max(1, int(np.round(np.sqrt(n_features)))-1)
        elif self.max_features == 'sqrt-2':
            max_features = max(1, int(np.round(np.sqrt(n_features)))-2)
        elif self.max_features == 'sqrt+1':
            max_features = int(np.round(np.sqrt(n_features))) + 1
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = self.max_features

        return max_features

    def bootstrap_sampling(self, X, y, sample_weight=None):
        """
        Carry out the bootstrap sampling

        Args:
            X (np.array): feature vectors
            y (np.array): the target labels

        Returns:
            np.array, np.array: the bootstrap sampled dataset
        """

        raise ValueError("bootstrap_sampling of base class called")

    def flip(self, X, idx=None):
        """
        Determine and record the flipping

        Args:
            X (np.array): the feature vectors

        Returns:
            int/np.array: the flipping multiplier
        """
        if self.flipping is None:
            self.flippings_.append(1)
        elif self.flipping == 'coordinate':
            self.flippings_.append(self.random_state.choice([-1, 1], X.shape[1], replace=True))
        elif self.flipping == 'full':
            if idx / self.n_estimators < 0.5:
                self.flippings_.append(1.0)
            else:
                self.flippings_.append(-1.0)

        return self.flippings_[-1]

    def fit_1(self, X, y):
        """
        Fitting with 1 job

        Args:
            X (np.array): the feature vectors
            y (np.array): the target labels

        Returns:
            obj: the fitted estimator object
        """
        _ = X
        _ = y
        raise ValueError("fit_1 method of base class called")

    def fit(self, X, y, sample_weight=None):
        """
        Fit the estimator

        Args:
            X (np.array): the feature vectors
            y (np.array): the target labels
            sample_weight (np.array): the sample weights

        Returns:
            obj: the fitted estimator object
        """
        self.estimators_ = []
        self.flippings_ = []

        if self.n_jobs == 1:
            return self.fit_1(X, y, sample_weight)

        params = self.get_params()
        params['n_jobs'] = 1
        params['n_estimators'] = int(self.n_estimators/self.n_jobs)

        forests = Parallel(n_jobs=self.n_jobs)(delayed(fit_forest)(params, X, y, self.__class__, sample_weight) for _ in range(self.n_jobs))

        for forest in forests:
            self.estimators_.extend(forest.estimators_)
            self.flippings_.extend(forest.flippings_)

        return self

class FlippingRandomForestClassifier2(FlippingBaggingBase, ClassifierMixin):
    """
    Flipping classification forest
    """
    def __init__(self,
                    n_estimators=100,
                    bootstrap=True,
                    bootstrap_features=False,
                    min_samples_leaf=1,
                    max_depth=None,
                    max_features='sqrt',
                    splitter='best',
                    flipping=None,
                    n_jobs=1,
                    random_state=None):
        """
        Constructor of the base class

        Args:
            n_estimators (int): the number of estimators
            bootstrap (bool): whether to do bootstrap sampling
            bootstrap_features (bool): whether to bootstrap the features
            min_samples_leaf (int): the minimum number of samples on leaf nodes
            max_depth (int): the maximum depth
            max_features (int/str): the number of features to test in nodes ('sqrt'/'auto')
            splitter (str): the splitter to be used ('random'/'best')
            n_jobs (int): the number of training jobs
            flipping (str): None/'coordinate'/'full' - the mode of flipping
            random_state (None/int/np.random.RandomState): the random state to be used
        """
        FlippingBaggingBase.__init__(self,
                                        n_estimators=n_estimators,
                                        bootstrap=bootstrap,
                                        bootstrap_features=bootstrap_features,
                                        min_samples_leaf=min_samples_leaf,
                                        max_depth=max_depth,
                                        max_features=max_features,
                                        splitter=splitter,
                                        flipping=flipping,
                                        n_jobs=n_jobs,
                                        random_state=random_state)
        ClassifierMixin.__init__(self)

    def bootstrap_sampling(self, X, y, sample_weight=None):
        """
        Carry out the bootstrap sampling

        Args:
            X (np.array): feature vectors
            y (np.array): the target labels

        Returns:
            np.array, np.array: the bootstrap sampled dataset
        """
        if self.bootstrap:
            class_labels = np.unique(y)

            Xs = []
            ys = []
            sws = []

            for class_label in class_labels:
                X_label = X[y == class_label]

                mask = self.random_state.choice(X_label.shape[0], X_label.shape[0], replace=True)

                Xs.append(X_label[mask])
                ys.append(np.repeat(class_label, X_label.shape[0]))

                if sample_weight is not None:
                    sw_label = sample_weight[y == class_label]
                    sws.append(sw_label[mask])

            if sample_weight is not None:
                sample_weight = np.hstack(sws)

            X = np.vstack(Xs)
            y = np.hstack(ys)

        return X, y, sample_weight


    def fit_1(self, X, y, sample_weight=None):
        """
        Fitting with 1 job

        Args:
            X (np.array): the feature vectors
            y (np.array): the target labels

        Returns:
            obj: the fitted estimator object
        """
        max_features = self.determine_max_features(X.shape[1])

        self.estimators_ = []
        self.flippings_ = []

        for idx in range(self.n_estimators):
            X_bs, y_bs, sw_bs = self.bootstrap_sampling(X, y, sample_weight)

            flipping = self.flip(X, idx)

            estimator = DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf,
                                                max_features=max_features,
                                                splitter=self.splitter,
                                                random_state=self._random_state_init)

            estimator.fit(X_bs*flipping, y_bs, sample_weight=sw_bs)

            self.estimators_.append(estimator)

        return self

    def predict_proba(self, X):
        """
        Predicts the probabilities

        Args:
            X (np.array): feature vectors

        Returns:
            np.array: the probabilities of the classes
        """
        results = []
        for idx, est in enumerate(self.estimators_):
            results.append(est.predict_proba(X*self.flippings_[idx])[:, 1])

        results = np.vstack(results)
        results = np.mean(results, axis=0)

        return np.vstack([1.0 - results, results]).T

    def predict(self, X):
        """
        Predicts class labels

        Args:
            X (np.array): the feature vectors

        Returns:
            np.array: the class labels
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)

class FlippingRandomForestRegressor2(FlippingBaggingBase, RegressorMixin):
    """
    The flipping forest regressor
    """
    def __init__(self,
                    n_estimators=100,
                    bootstrap=True,
                    bootstrap_features=False,
                    min_samples_leaf=1,
                    max_depth=None,
                    max_features=1.0,
                    splitter='best',
                    flipping=None,
                    n_jobs=1,
                    random_state=None):
        """
        Constructor of the base class

        Args:
            n_estimators (int): the number of estimators
            bootstrap (bool): whether to do bootstrap sampling
            bootstrap_features (bool): whether to bootstrap the features
            min_samples_leaf (int): the minimum number of samples on leaf nodes
            max_depth (int): the maximum depth
            max_features (int/str): the number of features to test in nodes ('sqrt'/'auto')
            splitter (str): the splitter to be used ('random'/'best')
            n_jobs (int): the number of training jobs
            flipping (str): None/'coordinate'/'full' - the mode of flipping
            random_state (None/int/np.random.RandomState): the random state to be used
        """

        FlippingBaggingBase.__init__(self,
                                        n_estimators=n_estimators,
                                        bootstrap=bootstrap,
                                        bootstrap_features=bootstrap_features,
                                        min_samples_leaf=min_samples_leaf,
                                        max_depth=max_depth,
                                        max_features=max_features,
                                        splitter=splitter,
                                        flipping=flipping,
                                        n_jobs=n_jobs,
                                        random_state=random_state)
        RegressorMixin.__init__(self)

    def bootstrap_sampling(self, X, y, sample_weight=None):
        """
        Carry out the bootstrap sampling

        Args:
            X (np.array): feature vectors
            y (np.array): the target labels

        Returns:
            np.array, np.array: the bootstrap sampled dataset
        """
        if self.bootstrap:
            mask = np.random.choice(X.shape[0], X.shape[0], replace=True)

            X = X[mask]
            y = y[mask]

            if sample_weight is not None:
                sample_weight = sample_weight[mask]

        return X, y, sample_weight


    def fit_1(self, X, y, sample_weight=None):
        """
        Fitting with 1 job

        Args:
            X (np.array): the feature vectors
            y (np.array): the target labels

        Returns:
            obj: the fitted estimator object
        """

        max_features = self.determine_max_features(X.shape[1])

        self.estimators_ = []
        self.flippings_ = []

        for idx in range(self.n_estimators):
            X_bs, y_bs, sw_bs = self.bootstrap_sampling(X, y, sample_weight)

            flipping = self.flip(X, idx)

            estimator = DecisionTreeRegressor(min_samples_leaf=self.min_samples_leaf,
                                                max_features=max_features,
                                                splitter=self.splitter,
                                                random_state=self._random_state_init)

            estimator.fit(X_bs*flipping, y_bs, sample_weight=sw_bs)

            self.estimators_.append(estimator)

        return self

    def predict(self, X):
        """
        Predicts regressed values

        Args:
            X (np.array): the feature vectors

        Returns:
            np.array: the regressed values
        """

        results = []
        for idx, est in enumerate(self.estimators_):
            results.append(est.predict(X*self.flippings_[idx]))

        results = np.vstack(results)
        results = np.mean(results, axis=0)

        return results
