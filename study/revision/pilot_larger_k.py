"""
Pilot experiment: does a real re-fit at larger K actually push the near-miss cells over the
significance line (under the corrected resampled t-test), as the formula-only projection
suggested? Uses the SAME selected hyperparameters as the existing study (isolating the K effect
from any hyperparameter-selection questions, which are a separate, later validation item).

Targets the near-miss cases found in the corrected-test re-analysis:
  - rfr / plastic   (<= vs <)      -- already significant at k=5, sanity check
  - dtr / plastic   (<= vs <)      -- p=0.091 at k=5
  - rfc / bupa      (<= vs <)      -- p=0.446 at k=5
  - rfr / wankara   (<= vs <)      -- p=0.449 at k=5
  - rfr / excitation_current (< vs avg_half) -- p=0.427 at k=5

n_repeats=100 (per the earlier analysis, the corrected-t floor is already >99% saturated by
n_repeats=100, so this loses essentially no power vs 400 while cutting compute ~4x).
"""
import sys
import os
import ast
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias')
sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias/study')
os.chdir('/home/gykovacs/workspaces/good/conditioning_bias/study')

from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, r2_score

from conditioning_bias import (
    OperatorDecisionTreeClassifier,
    OperatorDecisionTreeRegressor,
    OperatorRandomForestClassifier,
    OperatorRandomForestRegressor,
)
from datasets import binclas_datasets, regr_datasets
from scipy.stats import t as t_dist

RANDOM_SEED = 5
N_REPEATS = 100

def corrected_p(diffs, k):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean_d = np.mean(diffs)
    var_d = np.var(diffs, ddof=1)
    if var_d == 0:
        return 1.0, mean_d
    corrected_var = var_d * (1.0 / n + 1.0 / (k - 1))
    tstat = mean_d / np.sqrt(corrected_var)
    p = 2 * (1 - t_dist.cdf(np.abs(tstat), n - 1))
    return p, mean_d


def get_params(estimator_key, name):
    df = pd.read_csv(f'data/params_{estimator_key}.csv')
    row = df[df['name'] == name].iloc[0]
    params = eval(row['params'], {'np': np})
    params.pop('mode', None)
    return params


CLASSIFIER_CLASSES = {'dtc': OperatorDecisionTreeClassifier, 'rfc': OperatorRandomForestClassifier}
REGRESSOR_CLASSES = {'dtr': OperatorDecisionTreeRegressor, 'rfr': OperatorRandomForestRegressor}


def load_dataset(name, is_classification):
    table = binclas_datasets if is_classification else regr_datasets
    row = table[table['name'] == name].iloc[0]
    data = row['data_loader_function']()
    return data['data'], data['target']


def run_classification(estimator_key, name, k):
    X, y = load_dataset(name, True)
    params = get_params(estimator_key, name)
    est_cls = CLASSIFIER_CLASSES[estimator_key]

    random_state = np.random.RandomState(RANDOM_SEED)
    mask = np.arange(X.shape[0])
    random_state.shuffle(mask)
    X, y = X[mask], y[mask]

    n_splits = k
    if n_splits > np.sum(y):
        n_splits = int(np.sum(y))
    validator = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=N_REPEATS, random_state=RANDOM_SEED)

    diffs = []
    for train, test in validator.split(X, y, y):
        clf = est_cls(**(params | {'random_state': RANDOM_SEED, 'mode': '<='}))
        clf.fit(X[train], y[train])
        clf.set_mode('<=')
        p_leq = clf.predict_proba(X[test])[:, 1]
        clf.set_mode('<')
        p_l = clf.predict_proba(X[test])[:, 1]
        auc_leq = roc_auc_score(y[test], p_leq)
        auc_l = roc_auc_score(y[test], p_l)
        diffs.append(auc_leq - auc_l)
    return np.array(diffs)


def run_regression(estimator_key, name, k, pair=('<=', '<')):
    X, y = load_dataset(name, False)
    params = get_params(estimator_key, name)
    est_cls = REGRESSOR_CLASSES[estimator_key]

    validator = RepeatedKFold(n_splits=k, n_repeats=N_REPEATS, random_state=RANDOM_SEED)

    a_mode, b_mode = pair
    diffs = []
    for train, test in validator.split(X):
        reg = est_cls(**(params | {'random_state': RANDOM_SEED, 'mode': '<='}))
        reg.fit(X[train], y[train])
        reg.set_mode(a_mode)
        pred_a = reg.predict(X[test])
        reg.set_mode(b_mode)
        pred_b = reg.predict(X[test])
        r2_a = r2_score(y[test], pred_a)
        r2_b = r2_score(y[test], pred_b)
        diffs.append(r2_a - r2_b)
    return np.array(diffs)


if __name__ == '__main__':
    cases = [
        ('rfr', 'plastic', 'reg', ('<=', '<')),
        ('dtr', 'plastic', 'reg', ('<=', '<')),
        ('rfc', 'bupa', 'clf', None),
        ('rfr', 'wankara', 'reg', ('<=', '<')),
        ('rfr', 'excitation_current', 'reg', ('<', 'avg_half')),
    ]

    for estimator_key, name, kind, pair in cases:
        print(f"\n{'='*80}\n{estimator_key} / {name}  (k=5 baseline included for reference)")
        for k in [5, 10, 20, 30]:
            if kind == 'clf':
                diffs = run_classification(estimator_key, name, k)
            else:
                diffs = run_regression(estimator_key, name, k, pair)
            p, mean_d = corrected_p(diffs, k)
            print(f"  k={k:2d} (n_repeats={N_REPEATS}, n={len(diffs)}): "
                  f"mean_diff={mean_d:.3e}  corrected_p={p:.4f}  {'*' if p < 0.05 else ''}")
