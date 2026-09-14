"""
Combined rerun of the K-sweep (real_data_ksweep.py), extended to:
  1. save absolute per-operator mean scores (mean_leq, mean_lt, mean_avg, mean_select), not just
     differences -- needed to show absolute AUC/r2 values in Tables 2/3 (reviewer 4, point 6),
     and needed at the SAME K as the significance test since the point estimate itself is not
     K-invariant (verified: e.g. bupa's <=_vs_< mean_diff changes sign between K=5 and K=10+).
  2. add a fourth mode: 'select', a training-in-sample proxy for CV-based operator selection
     (reviewer 4, point 1) -- at each fold, after fitting, score both '<=' and '<' on the
     TRAINING data itself (in-sample) and report the test-fold score of whichever operator won
     in-sample. No extra fitting needed (same one fit per fold as before), just extra predict
     calls, so this does not multiply the already-expensive fitting cost.

New comparison pairs added to the existing '<=_vs_<' and 'avg_vs_worst':
  - 'select_vs_avg':   does operator-selection beat the proposed averaging method?
  - 'select_vs_worst': does operator-selection at least avoid the worse individual operator?
  - 'select_hit_rate': descriptive diagnostic -- how often does the in-sample-selected operator
    actually match the operator that turns out better on the held-out test fold (vs. 50% chance)?
"""
import os
import sys
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
K_VALUES = [5, 10, 15, 20]
MIN_TEST_FOLD_SIZE = 3
EXCLUDED_REGR = ['o-ring', 'diabetes']
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


def make_rows(estimator_key, name, k, per_fold):
    """
    per_fold: dict of lists, one entry per fold, keys leq/lt/avg/select (test scores).

    NOTE on select_hit_rate: restricted to folds where the two operators actually differ on the
    TEST set (leq != lt). Including tied folds is a bug we caught and fixed: both "which operator
    was selected" (train-side >=) and "which operator was actually better" (test-side >=) use the
    same tie-breaking convention, so on tied folds (the common case) they trivially "match" by
    construction regardless of any real selection skill, spuriously inflating the apparent hit
    rate. Restricting to non-tied folds removes this artifact. This is still a secondary,
    descriptive diagnostic (filtering to a data-dependent subset of folds is an informal use of
    the corrected-t machinery, unlike select_vs_avg/select_vs_worst below which use ALL folds
    unfiltered and are the primary, rigorously valid comparisons).
    """
    leq = np.array(per_fold['leq'])
    lt = np.array(per_fold['lt'])
    avg = np.array(per_fold['avg'])
    select = np.array(per_fold['select'])
    hit = np.array(per_fold['hit'])  # 1 if selected op matches the test-better op, else 0

    worst = np.minimum(leq, lt)
    non_tied = leq != lt
    n_valid = int(non_tied.sum())

    pairs = {
        '<=_vs_<': (leq - lt, k),
        'avg_vs_worst': (avg - worst, k),
        'select_vs_avg': (select - avg, k),
        'select_vs_worst': (select - worst, k),
    }
    rows = []
    for pair, (diffs, kk) in pairs.items():
        p, m = corrected_p(diffs, kk)
        rows.append({
            'estimator': estimator_key, 'name': name, 'k': k, 'pair': pair,
            'mean_diff': m, 'corrected_p': p, 'n_valid': len(diffs),
            'mean_leq': leq.mean(), 'mean_lt': lt.mean(),
            'mean_avg': avg.mean(), 'mean_select': select.mean(),
        })

    if n_valid >= 2:
        p, m = corrected_p(hit[non_tied] - 0.5, k)
        m = m + 0.5
    else:
        p, m = np.nan, np.nan
    rows.append({
        'estimator': estimator_key, 'name': name, 'k': k, 'pair': 'select_hit_rate',
        'mean_diff': m, 'corrected_p': p, 'n_valid': n_valid,
        'mean_leq': leq.mean(), 'mean_lt': lt.mean(),
        'mean_avg': avg.mean(), 'mean_select': select.mean(),
    })
    return rows


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
        per_fold = {'leq': [], 'lt': [], 'avg': [], 'select': [], 'hit': []}
        for train, test in validator.split(X, y, y):
            clf = est_cls(**(params | {'random_state': RANDOM_SEED, 'mode': '<='}))
            clf.fit(X[train], y[train])

            clf.set_mode('<=')
            p_leq_test = clf.predict_proba(X[test])[:, 1]
            p_leq_train = clf.predict_proba(X[train])[:, 1]
            clf.set_mode('<')
            p_lt_test = clf.predict_proba(X[test])[:, 1]
            p_lt_train = clf.predict_proba(X[train])[:, 1]
            clf.set_mode(avg_mode)
            p_avg_test = clf.predict_proba(X[test])[:, 1]

            auc_leq_test = roc_auc_score(y[test], p_leq_test)
            auc_lt_test = roc_auc_score(y[test], p_lt_test)
            auc_avg_test = roc_auc_score(y[test], p_avg_test)
            auc_leq_train = roc_auc_score(y[train], p_leq_train)
            auc_lt_train = roc_auc_score(y[train], p_lt_train)

            select_is_leq = auc_leq_train >= auc_lt_train
            auc_select_test = auc_leq_test if select_is_leq else auc_lt_test
            test_better_is_leq = auc_leq_test >= auc_lt_test
            hit = float(select_is_leq == test_better_is_leq)

            per_fold['leq'].append(auc_leq_test)
            per_fold['lt'].append(auc_lt_test)
            per_fold['avg'].append(auc_avg_test)
            per_fold['select'].append(auc_select_test)
            per_fold['hit'].append(hit)

        rows.extend(make_rows(estimator_key, name, k, per_fold))
    return rows


def run_regression(estimator_key, name, X, y, params):
    est_cls = REGRESSOR_CLASSES[estimator_key]
    avg_mode = AVG_MODE[estimator_key]

    rows = []
    for k in effective_k_values(X.shape[0]):
        validator = RepeatedKFold(n_splits=k, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
        per_fold = {'leq': [], 'lt': [], 'avg': [], 'select': [], 'hit': []}
        for train, test in validator.split(X):
            reg = est_cls(**(params | {'random_state': RANDOM_SEED, 'mode': '<='}))
            reg.fit(X[train], y[train])

            reg.set_mode('<=')
            pred_leq_test = reg.predict(X[test])
            pred_leq_train = reg.predict(X[train])
            reg.set_mode('<')
            pred_lt_test = reg.predict(X[test])
            pred_lt_train = reg.predict(X[train])
            reg.set_mode(avg_mode)
            pred_avg_test = reg.predict(X[test])

            r2_leq_test = r2_score(y[test], pred_leq_test)
            r2_lt_test = r2_score(y[test], pred_lt_test)
            r2_avg_test = r2_score(y[test], pred_avg_test)
            r2_leq_train = r2_score(y[train], pred_leq_train)
            r2_lt_train = r2_score(y[train], pred_lt_train)

            select_is_leq = r2_leq_train >= r2_lt_train
            r2_select_test = r2_leq_test if select_is_leq else r2_lt_test
            test_better_is_leq = r2_leq_test >= r2_lt_test
            hit = float(select_is_leq == test_better_is_leq)

            per_fold['leq'].append(r2_leq_test)
            per_fold['lt'].append(r2_lt_test)
            per_fold['avg'].append(r2_avg_test)
            per_fold['select'].append(r2_select_test)
            per_fold['hit'].append(hit)

        rows.extend(make_rows(estimator_key, name, k, per_fold))
    return rows


def get_regr_dataset_pool():
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
    out_path = 'data/ksweep_results_v3.csv'

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
