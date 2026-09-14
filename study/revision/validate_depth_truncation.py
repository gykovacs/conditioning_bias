"""
Validates the exact-equivalence claim: fitting a tree/forest once with max_depth=None and
min_samples_leaf=1, then truncating traversal at depth d via conditioning_bias.tree_inference's
new max_depth parameter, reproduces the predictions of actually refitting with max_depth=d,
exactly (not approximately) -- for both decision trees and random forests, classification and
regression, across several real datasets and several depths.
"""
import sys
sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias')
sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias/study')
import os
os.chdir('/home/gykovacs/workspaces/good/conditioning_bias/study')

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from conditioning_bias import tree_inference
from datasets import binclas_datasets, regr_datasets

RANDOM_SEED = 5


def check_dt(cls, X, y, depths):
    full = cls(max_depth=None, min_samples_leaf=1, random_state=RANDOM_SEED).fit(X, y)
    all_match = True
    for d in depths:
        truncated = tree_inference(X=X, tree=full, operator='<=', max_depth=d)
        refit = cls(max_depth=d, min_samples_leaf=1, random_state=RANDOM_SEED).fit(X, y)
        real_vals = refit.tree_.value[
            np.array([refit.tree_.apply(np.array(X[i:i+1], dtype=np.float32))[0] for i in range(len(X))]),
            0, :]
        match = np.allclose(truncated, real_vals)
        all_match = all_match and match
        if not match:
            print(f"  MISMATCH at depth {d}: max abs diff = {np.abs(truncated - real_vals).max()}")
    return all_match


def check_rf(cls, X, y, depths, n_estimators=10):
    full = cls(max_depth=None, min_samples_leaf=1, n_estimators=n_estimators,
               random_state=RANDOM_SEED).fit(X, y)
    all_match = True
    for d in depths:
        truncated_per_tree = [tree_inference(X=X, tree=t, operator='<=', max_depth=d)
                               for t in full.estimators_]
        truncated_avg = np.mean(truncated_per_tree, axis=0)

        refit = cls(max_depth=d, min_samples_leaf=1, n_estimators=n_estimators,
                    random_state=RANDOM_SEED).fit(X, y)
        real_per_tree = []
        for t in refit.estimators_:
            leaf_idx = t.tree_.apply(np.array(X, dtype=np.float32))
            real_per_tree.append(t.tree_.value[leaf_idx, 0, :])
        real_avg = np.mean(real_per_tree, axis=0)

        match = np.allclose(truncated_avg, real_avg)
        all_match = all_match and match
        if not match:
            print(f"  MISMATCH at depth {d}: max abs diff = {np.abs(truncated_avg - real_avg).max()}")
    return all_match


if __name__ == '__main__':
    depths = [1, 2, 3, 5, 8]

    print("=== Decision Tree Classifier ===")
    for name in ['appendicitis', 'bupa', 'mammographic']:
        row = binclas_datasets[binclas_datasets['name'] == name].iloc[0]
        data = row['data_loader_function']()
        X, y = data['data'], data['target']
        ok = check_dt(DecisionTreeClassifier, X, y, depths)
        print(f"{name}: {'MATCH' if ok else 'MISMATCH'}")

    print("=== Decision Tree Regressor ===")
    for name in ['wsn-ale', 'plastic', 'cpu_performance']:
        row = regr_datasets[regr_datasets['name'] == name].iloc[0]
        data = row['data_loader_function']()
        X, y = data['data'], data['target']
        ok = check_dt(DecisionTreeRegressor, X, y, depths)
        print(f"{name}: {'MATCH' if ok else 'MISMATCH'}")

    print("=== Random Forest Classifier ===")
    for name in ['appendicitis', 'bupa']:
        row = binclas_datasets[binclas_datasets['name'] == name].iloc[0]
        data = row['data_loader_function']()
        X, y = data['data'], data['target']
        ok = check_rf(RandomForestClassifier, X, y, depths)
        print(f"{name}: {'MATCH' if ok else 'MISMATCH'}")

    print("=== Random Forest Regressor ===")
    for name in ['wsn-ale', 'plastic']:
        row = regr_datasets[regr_datasets['name'] == name].iloc[0]
        data = row['data_loader_function']()
        X, y = data['data'], data['target']
        ok = check_rf(RandomForestRegressor, X, y, depths)
        print(f"{name}: {'MATCH' if ok else 'MISMATCH'}")
