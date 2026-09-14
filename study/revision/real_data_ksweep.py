"""
Core protocol, real-data piece: K-sweep across ALL 40 datasets x 4 estimators, using the
corrected resampled t-test throughout, reusing the existing selected hyperparameters
(study/data/params_{label}.csv -- nested CV is a separate, later stretch goal).

For each dataset x estimator x K in {5,10,15,20} (capped per-dataset so test-fold size >= 3
records -- see floor rule discussed with the user), runs n_repeats=100 (validated as sufficient
to saturate the corrected-test floor) repeated K-fold CV, and computes the corrected resampled
t-test p-value for two comparisons: '<=' vs '<', and avg vs worst(<=, <).

Output: a single long-format CSV (data/ksweep_results.csv) with one row per
(estimator, dataset, K, pair) -- easy to aggregate into the meta-analysis afterward.
"""
import os
import sys
import ast
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias')
sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias/study')
os.chdir('/home/gykovacs/workspaces/good/conditioning_bias/study')

from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, r2_score

from conditioning_bias import (
    OperatorDecisionTreeClassifier, OperatorDecisionTreeRegressor,
    OperatorRandomForestClassifier, OperatorRandomForestRegressor,
)
from datasets import binclas_datasets, regr_datasets

RANDOM_SEED = 5
N_REPEATS = 100
K_VALUES = [10, 15, 20]  # k=5 reused from existing 400-repeat study data, see reuse_k5.py
MIN_TEST_FOLD_SIZE = 3
EXCLUDED_REGR = ['o-ring', 'diabetes']  # replaced -- too small for stable 5-fold CV selection
NEW_REGR = ['concrete', 'boom_bikes']

CLASSIFIER_CLASSES = {'dtc': OperatorDecisionTreeClassifier, 'rfc': OperatorRandomForestClassifier}
REGRESSOR_CLASSES = {'dtr': OperatorDecisionTreeRegressor, 'rfr': OperatorRandomForestRegressor}
AVG_MODE = {'dtc': 'avg_full', 'rfc': 'avg_half', 'dtr': 'avg_full', 'rfr': 'avg_half'}


def corrected_p(diffs, k):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean_d = np.mean(diffs)
    var_d = np.var(diffs, ddof=1)
    if var_d == 0:
        return (1.0 if mean_d == 0 else 0.0), mean_d
    corrected_var = var_d * (1.0 / n + 1.0 / (k - 1))
    tstat = mean_d / np.sqrt(corrected_var)
    p = 2 * (1 - t_dist.cdf(np.abs(tstat), n - 1))
    return p, mean_d


def get_params(estimator_key, name):
    df = pd.read_csv(f'data/params_{estimator_key}.csv')
    row = df[df['name'] == name].iloc[0]
    params = eval(row['params'], {'np': np})
    params.pop('mode', None)
    params.pop('random_state', None)
    return params


def effective_k_values(n_samples, min_minority=None):
    ks = []
    for k in K_VALUES:
        cap = k
        if min_minority is not None:
            cap = min(cap, min_minority)
        if n_samples / cap >= MIN_TEST_FOLD_SIZE and cap >= 2:
            ks.append(cap)
    return sorted(set(ks))


def run_classification(estimator_key, name, X, y, params):
    est_cls = CLASSIFIER_CLASSES[estimator_key]
    avg_mode = AVG_MODE[estimator_key]

    random_state = np.random.RandomState(RANDOM_SEED)
    mask = np.arange(X.shape[0])
    random_state.shuffle(mask)
    X, y = X[mask], y[mask]

    n_minority = int(min(np.sum(y == c) for c in np.unique(y)))
    rows = []
    for k in effective_k_values(X.shape[0], min_minority=n_minority):
        validator = RepeatedStratifiedKFold(n_splits=k, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
        diffs_lt, diffs_avg = [], []
        for train, test in validator.split(X, y, y):
            clf = est_cls(**(params | {'random_state': RANDOM_SEED, 'mode': '<='}))
            clf.fit(X[train], y[train])
            clf.set_mode('<=')
            p_leq = clf.predict_proba(X[test])[:, 1]
            clf.set_mode('<')
            p_lt = clf.predict_proba(X[test])[:, 1]
            clf.set_mode(avg_mode)
            p_avg = clf.predict_proba(X[test])[:, 1]

            auc_leq = roc_auc_score(y[test], p_leq)
            auc_lt = roc_auc_score(y[test], p_lt)
            auc_avg = roc_auc_score(y[test], p_avg)

            diffs_lt.append(auc_leq - auc_lt)
            diffs_avg.append(auc_avg - min(auc_leq, auc_lt))

        p1, m1 = corrected_p(diffs_lt, k)
        p2, m2 = corrected_p(diffs_avg, k)
        rows.append({'estimator': estimator_key, 'name': name, 'k': k, 'pair': '<=_vs_<',
                     'mean_diff': m1, 'corrected_p': p1})
        rows.append({'estimator': estimator_key, 'name': name, 'k': k, 'pair': 'avg_vs_worst',
                     'mean_diff': m2, 'corrected_p': p2})
    return rows


def run_regression(estimator_key, name, X, y, params):
    est_cls = REGRESSOR_CLASSES[estimator_key]
    avg_mode = AVG_MODE[estimator_key]

    rows = []
    for k in effective_k_values(X.shape[0]):
        validator = RepeatedKFold(n_splits=k, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
        diffs_lt, diffs_avg = [], []
        for train, test in validator.split(X):
            reg = est_cls(**(params | {'random_state': RANDOM_SEED, 'mode': '<='}))
            reg.fit(X[train], y[train])
            reg.set_mode('<=')
            pred_leq = reg.predict(X[test])
            reg.set_mode('<')
            pred_lt = reg.predict(X[test])
            reg.set_mode(avg_mode)
            pred_avg = reg.predict(X[test])

            r2_leq = r2_score(y[test], pred_leq)
            r2_lt = r2_score(y[test], pred_lt)
            r2_avg = r2_score(y[test], pred_avg)

            diffs_lt.append(r2_leq - r2_lt)
            diffs_avg.append(r2_avg - min(r2_leq, r2_lt))

        p1, m1 = corrected_p(diffs_lt, k)
        p2, m2 = corrected_p(diffs_avg, k)
        rows.append({'estimator': estimator_key, 'name': name, 'k': k, 'pair': '<=_vs_<',
                     'mean_diff': m1, 'corrected_p': p1})
        rows.append({'estimator': estimator_key, 'name': name, 'k': k, 'pair': 'avg_vs_worst',
                     'mean_diff': m2, 'corrected_p': p2})
    return rows


def get_regr_dataset_pool():
    """Original 18 continuing regression datasets + concrete/boom_bikes replacing o-ring/diabetes."""
    import common_datasets.regression as regr
    continuing = regr_datasets[~regr_datasets['name'].isin(EXCLUDED_REGR)][
        ['name', 'data_loader_function']]
    all_loaders = regr.get_filtered_data_loaders(n_bounds=(1, 1_000_000), n_from_phenotypes=1)
    summary = regr.get_summary_pdf()
    summary = summary[summary['data_loader_function'].isin(all_loaders)]
    new_rows = summary[summary['name'].isin(NEW_REGR)][['name', 'data_loader_function']]
    return pd.concat([continuing, new_rows], ignore_index=True)


if __name__ == '__main__':
    all_rows = []
    out_path = 'data/ksweep_results.csv'

    for _, row in binclas_datasets.iterrows():
        name = row['name']
        data = row['data_loader_function']()
        X, y = data['data'], data['target']
        for estimator_key in ['dtc', 'rfc']:
            params = get_params(estimator_key, name)
            print(f"{estimator_key} / {name}", flush=True)
            rows = run_classification(estimator_key, name, X, y, params)
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(out_path, index=False)

    regr_pool = get_regr_dataset_pool()
    for _, row in regr_pool.iterrows():
        name = row['name']
        data = row['data_loader_function']()
        X, y = data['data'], data['target']
        for estimator_key in ['dtr', 'rfr']:
            params = get_params(estimator_key, name)
            print(f"{estimator_key} / {name}", flush=True)
            rows = run_regression(estimator_key, name, X, y, params)
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(out_path, index=False)

    print("DONE", flush=True)
