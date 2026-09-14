"""
Standalone synthetic-data validation experiment.

Independent of the real-data nested-CV pipeline in full_experiment.py: draws data from a known
generative process with a controllable discrete/equidistant feature (synthetic.py), so that a
fresh, effectively unlimited, i.i.d. test set can be drawn for every replicate. This sidesteps the
whole repeated-CV non-independence issue that motivates the corrected significance test elsewhere
in the study -- replicate scores here are genuinely independent by construction, so an ordinary
paired test is valid without any correction.

Protocol per replicate:
  1. draw a fresh training set (fixed size) from the generative process
  2. fit the tree/forest once, with a fixed hyperparameter (not tuned per replicate -- there is no
     hyperparameter-selection leakage to control for here, since the generative process, not a
     finite real dataset, defines ground truth)
  3. draw a large, fresh, independent i.i.d. test set and score <=, <, and the proposed average on it
  4. record the two score differences of interest for this replicate: <= vs <, and avg vs worst(<=,<)

Repeated for N_REPLICATES independent replicates per (estimator, sweep, value) configuration, over
two sweeps: discrete-domain granularity (at a fixed depth) and tree depth (at a fixed domain size).
Covers all four estimator/problem-type combinations used in the real-data study (dtc, rfc, dtr,
rfr). Parallelized across replicates at 80% of available CPUs.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias')
sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias/study/revision')
os.chdir('/home/gykovacs/workspaces/good/conditioning_bias/study/revision')

from synthetic import make_conditioning_bias_regression, make_conditioning_bias_classification
from conditioning_bias import (
    OperatorDecisionTreeRegressor, OperatorRandomForestRegressor,
    OperatorDecisionTreeClassifier, OperatorRandomForestClassifier,
)
from sklearn.metrics import r2_score, roc_auc_score

N_JOBS = max(1, int(0.8 * os.cpu_count()))
N_TRAIN = 500
N_TEST_EVAL = 50_000  # still 100x N_TRAIN; the random-forest apply() path is the wall-time
                       # bottleneck at scale, so this trades some test-set precision for
                       # tractability rather than the (much cheaper) N_TRAIN or N_REPLICATES
N_REPLICATES = 100     # matches the R=100 power-saturation rationale used in full_experiment.py
RANDOM_SEED = 5

DOMAIN_SWEEP = [5, 11, 21, 51]
DEPTH_SWEEP = [3, 5, 8, 12]
FIXED_DEPTH = 8
FIXED_DOMAIN = 11

# estimator key -> (class, extra kwargs, averaging mode, is_classification)
# 'avg_full'/'avg_half' match the free-lunch averaging convention used throughout the paper: a
# single tree evaluates both operators (avg_full), a forest splits its trees in half between the
# two operators (avg_half), so random forest averaging costs nothing extra at prediction time.
ESTIMATORS = {
    'dtc': (OperatorDecisionTreeClassifier, {}, 'avg_full', True),
    'rfc': (OperatorRandomForestClassifier, {'n_estimators': 50}, 'avg_half', True),
    'dtr': (OperatorDecisionTreeRegressor, {}, 'avg_full', False),
    'rfr': (OperatorRandomForestRegressor, {'n_estimators': 50}, 'avg_half', False),
}


def _score(y_true, y_pred, is_classif):
    return roc_auc_score(y_true, y_pred) if is_classif else r2_score(y_true, y_pred)


def _predict(model, X, is_classif):
    return model.predict_proba(X)[:, 1] if is_classif else model.predict(X)


def _replicate(estimator_key, domain_size, depth, seed, is_classif):
    cls, kwargs, avg_mode, _ = ESTIMATORS[estimator_key]
    gen = make_conditioning_bias_classification if is_classif else make_conditioning_bias_regression

    X_train, y_train = gen(N_TRAIN, discrete_domain_size=domain_size, random_state=seed)
    X_test, y_test = gen(N_TEST_EVAL, discrete_domain_size=domain_size, random_state=seed + 500_000)

    model = cls(mode='<=', random_state=RANDOM_SEED, max_depth=depth, **kwargs)
    model.fit(X_train, y_train)

    model.set_mode('<=')
    s_leq = _score(y_test, _predict(model, X_test, is_classif), is_classif)
    model.set_mode('<')
    s_lt = _score(y_test, _predict(model, X_test, is_classif), is_classif)
    model.set_mode(avg_mode)
    s_avg = _score(y_test, _predict(model, X_test, is_classif), is_classif)

    worst = min(s_leq, s_lt)
    return {'leq_vs_lt': s_leq - s_lt, 'avg_vs_worst': s_avg - worst,
            's_leq': s_leq, 's_lt': s_lt, 's_avg': s_avg, 's_worst': worst}


def run_config(estimator_key, domain_size, depth, is_classif, seed0):
    out = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(_replicate)(estimator_key, domain_size, depth, seed0 * 100_000 + rep, is_classif)
        for rep in range(N_REPLICATES)
    )
    df = pd.DataFrame(out)
    rows = []
    pair_cols = {'leq_vs_lt': ('s_leq', 's_lt'), 'avg_vs_worst': ('s_avg', 's_worst')}
    for pair in ['leq_vs_lt', 'avg_vs_worst']:
        diffs = df[pair].values
        mean_d = float(diffs.mean())
        p = 1.0 if np.all(diffs == 0) else float(wilcoxon(diffs, zero_method='zsplit').pvalue)
        col_a, col_b = pair_cols[pair]
        rows.append({'estimator': estimator_key, 'domain_size': domain_size, 'depth': depth,
                     'pair': pair, 'mean_diff': mean_d, 'wilcoxon_p': p, 'n': len(diffs),
                     'mean_a': float(df[col_a].mean()), 'mean_b': float(df[col_b].mean())})
    return rows


if __name__ == '__main__':
    rows = []

    def process(estimator_key, domain_size, depth, sweep, seed0):
        is_classif = ESTIMATORS[estimator_key][3]
        t0 = time.time()
        cfg_rows = run_config(estimator_key, domain_size, depth, is_classif, seed0)
        for r in cfg_rows:
            r['sweep'] = sweep
        rows.extend(cfg_rows)
        print(f"{estimator_key} sweep={sweep} domain_size={domain_size} depth={depth} "
              f"({time.time()-t0:.1f}s)", flush=True)
        pd.DataFrame(rows).to_csv('../data/synthetic_results.csv', index=False)

    print(f"N_JOBS = {N_JOBS} (80% of {os.cpu_count()} cores)", flush=True)

    seed0 = 0
    for est in ESTIMATORS:
        for domain_size in DOMAIN_SWEEP:
            seed0 += 1
            process(est, domain_size, FIXED_DEPTH, 'domain', seed0)
        for depth in DEPTH_SWEEP:
            seed0 += 1
            process(est, FIXED_DOMAIN, depth, 'depth', seed0)

    print("DONE", flush=True)
