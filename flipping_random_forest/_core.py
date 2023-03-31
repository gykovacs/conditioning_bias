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
            'do_comparisons']

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
