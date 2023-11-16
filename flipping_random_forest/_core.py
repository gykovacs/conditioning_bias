"""
This module implements some core functionalities
"""

import copy

import tqdm

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

__all__ = ['grid_point_split_feature',
            'grid_point_splits',
            'mirror_tree',
            'evaluate',
            'evaluate_scenarios_regression',
            'evaluate_scenarios_classification',
            'do_comparisons',
            'RandomStateMixin',
            'FlippingBaggingBase',
            'fit_forest']

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


def grid_point_split_feature(tree, indicator, X, feature_idx, node_idx):
    """
    Determines the characteristics of splits at a certain node

    Args:
        tree (sklearn.tree.DecisionTree*): the fitted tree
        X (np.array): the feature vectors
        feature_idx (int): the feature being processed
        node_idx (int): the node idx

    Returns:
        bool, float, float, float, float: a flag indicating if the split is on
                a grid point, the grid constant, the lower and upper boundaries
                of the splitting interval, the threshold
    """
    left_idx = tree.tree_.children_left[node_idx]
    right_idx = tree.tree_.children_right[node_idx]

    left_samples = X[indicator[:, left_idx].astype(bool)]
    right_samples = X[indicator[:, right_idx].astype(bool)]

    lower = np.max(left_samples[:, feature_idx])
    upper = np.min(right_samples[:, feature_idx])

    assert lower <= tree.tree_.threshold[node_idx] <= upper, str(f"{lower, tree.tree_.threshold[node_idx], upper}")

    diff = tree.tree_.threshold[node_idx] - lower

    unique_feature_values = np.unique(X[:, feature_idx])
    tmp = np.sort(unique_feature_values)
    grid_constant = np.min(tmp[1:] - tmp[:-1])

    return ((diff / grid_constant) - np.floor(diff/grid_constant) == 0.0), grid_constant, lower, upper, tree.tree_.threshold[node_idx]

def grid_point_splits(tree, X, grid):
    """
    Returns the grid splits for a tree fitted to the data X with grid features indicated in grid

    Args:
        tree (sklearn.tree.DecisionTree*): the fitted tree
        X (np.array): the feature vectors
        grid (list(bool)): the indicator of grid features

    Returns:
        pd.DataFrame: a summary of grid splits
    """
    indicator = np.array(tree.decision_path(X).todense())
    features = tree.tree_.feature

    results = []

    for feature_idx in range(X.shape[1]):
        if grid[feature_idx]:
            for node_idx in np.where(features == feature_idx)[0]:
                grid_split, grid_constant, lower, upper, threshold = grid_point_split_feature(tree, indicator, X, feature_idx, node_idx)
                results.append([node_idx, feature_idx, grid_split, grid_constant, lower, upper, threshold, np.sum(features >= 0)])

    return pd.DataFrame(results, columns=['node_idx', 'feature_idx', 'grid_split', 'grid_constant', 'lower', 'upper', 'threshold', 'all_nodes'])

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

def evaluate(scenarios,
                compare,
                data_loaders,
                validator_params,
                score,
                random_state=None):
    results = []

    names_means = [f"{score}_mean_{scenario['name']}" for scenario in scenarios]
    names_scores = [f"{score}s_{scenario['name']}" for scenario in scenarios]
    names_comparisons = [f"p_{comparison[0]}_{comparison[1]}" for comparison in compare]

    for data_loader in data_loaders:
        dataset = data_loader()
        X = dataset['data']
        y = dataset['target']

        if score == 'r2':
            raw_scores = evaluate_scenarios_regression(scenarios, X, y, validator_params, random_state=random_state)
        elif score == 'auc':
            raw_scores = evaluate_scenarios_classification(scenarios, X, y, validator_params, random_state=random_state)

        mean_scores = [np.mean(raw_scores[scenario['name']]) for scenario in scenarios]
        scores = [raw_scores[scenario['name']] for scenario in scenarios]

        comparisons = do_comparisons(raw_scores, compare)

        results.append([dataset['name']] + mean_scores + scores + comparisons)

        tmp = pd.DataFrame([results[-1]], columns=['name'] + names_means + names_scores + names_comparisons)
        print(tmp[['name'] + names_means + names_comparisons])

    return pd.DataFrame(results, columns=['name'] + names_means + names_scores + names_comparisons)

def evaluate_scenarios_regression(scenarios, X, y, validator_params, random_state=None):
    r2s = {scenario['name']: [] for scenario in scenarios}

    if random_state is not None:
        random_state = np.random.RandomState(random_state)

    if isinstance(validator_params, dict):
        validator = RepeatedKFold(**validator_params)
        for train, test in tqdm.tqdm(validator.split(X, y)):
            random_seed = None
            if random_state is not None:
                random_seed = random_state.randint(validator_params['n_repeats'] * validator_params['n_splits'] * 100)

            X_train = X[train]
            X_test = X[test]
            y_train = y[train]
            y_test = y[test]

            for scenario in scenarios:
                estimator = scenario['estimator'](**scenario['estimator_params'], random_state=random_seed)
                estimator.fit(X_train * scenario['multiplier'], y_train)
                pred = estimator.predict(X_test * scenario['multiplier'])
                r2s[scenario['name']].append(r2_score(y_test, pred))
    else:
        for idx in tqdm.tqdm(range(validator_params)):
            random_seed = None
            if random_state is not None:
                random_seed = random_state.randint(10000)

            for scenario in scenarios:
                estimator = scenario['estimator'](**scenario['estimator_params'], random_state=random_seed)
                estimator.fit(X * scenario['multiplier'], y)
                pred = estimator.predict(X * scenario['multiplier'])
                r2s[scenario['name']].append(r2_score(y, pred))

    return r2s

def evaluate_scenarios_classification(scenarios, X, y, validator_params, random_state=None):
    aucs = {scenario['name']: [] for scenario in scenarios}

    if random_state is not None:
        random_state = np.random.RandomState(random_state)

    if isinstance(validator_params, dict):
        validator = RepeatedStratifiedKFold(**validator_params)
        for train, test in tqdm.tqdm(validator.split(X, y)):
            random_seed = None
            if random_state is not None:
                random_seed = random_state.randint(validator_params['n_repeats'] * validator_params['n_splits'] * 100)

            X_train = X[train]
            X_test = X[test]
            y_train = y[train]
            y_test = y[test]

            for scenario in scenarios:
                estimator = scenario['estimator'](**scenario['estimator_params'], random_state=random_seed)
                estimator.fit(X_train * scenario['multiplier'], y_train)
                pred = estimator.predict_proba(X_test * scenario['multiplier'])[:, 1]
                aucs[scenario['name']].append(roc_auc_score(y_test, pred))
    else:
        for idx in tqdm.tqdm(range(validator_params)):
            random_seed = None
            if random_state is not None:
                random_seed = random_state.randint(10000)

            for scenario in scenarios:
                estimator = scenario['estimator'](**scenario['estimator_params'], random_state=random_seed)
                estimator.fit(X * scenario['multiplier'], y)
                pred = estimator.predict_proba(X * scenario['multiplier'])[:, 1]
                aucs[scenario['name']].append(roc_auc_score(y, pred))

    return aucs

def do_comparisons(scores, compare):
    comparisons = []
    for comparison in compare:
        comparisons.append(wilcoxon(scores[comparison[0]],
                                    scores[comparison[1]],
                                    alternative='less',
                                    zero_method='zsplit').pvalue)

    return comparisons
