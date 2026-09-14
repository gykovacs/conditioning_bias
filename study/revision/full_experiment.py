"""
ONE script producing everything the paper needs, under the redesigned methodology: max_depth-only
regularization (min_samples_leaf dropped entirely, from every evaluation, not just the nested-CV
check), with hyperparameter selection done by growing one full-depth tree/forest per fold and
reading off every candidate depth via truncated traversal (conditioning_bias.apply/tree_inference's
max_depth parameter) instead of refitting per candidate -- valid because, per fold, any place
truncation and a real max_depth=d refit could disagree is necessarily a tied-optimal split (ties
are the only situation where sklearn's split selection depends on the RNG stream, and the RNG
stream is the only way depth-elsewhere-in-the-tree can affect a node's outcome), so the truncated
tree is an equally-valid max_depth=d model, not a worse one. See nested_cv.py's docstring and
study/revision/validate_depth_truncation.py for the full argument and the empirical check that
motivated it.

Produces, per (estimator, dataset):
  1. A reference max_depth (selection via CV on the FULL dataset -- descriptive only, matches
     what Table 1 has always reported, i.e. "what a practitioner tuning normally would pick";
     NOT used to evaluate performance, so it carries no leakage risk for the numbers that matter).
  2. rho / rho_k (lattice-aligned-threshold node proportions, fit-to-all-data and averaged across
     the many fold-fits produced along the way).
  3. Repeated random holdout (train_size=0.9), properly nested hyperparameter AND operator
     selection per split -- no leakage anywhere: '<=_vs_<', 'avg_vs_worst', 'select_vs_avg',
     'select_vs_worst', 'select_hit_rate' (the last restricted to splits where the operators
     genuinely differ on the held-out data, per the tie-breaking-convention bug fix documented in
     real_data_ksweep_v2.py). Replaces an earlier K-fold sweep design (K in {5,10,15,20}):
     train_size=0.9 gives n_test/n_train=1/9, identical to K=10's K-fold floor (1/(k-1)=1/9), so
     the corrected_p formula is reused unchanged with k=HOLDOUT_K_EQUIV=10 -- not an approximation,
     the two ratios are exactly equal, just realized via independent random splits (Nadeau &
     Bengio 2003's original setting for this correction) instead of a K-fold partition (Bouckaert &
     Frank 2004's adaptation of the same correction).

Parallelized across outer folds at 80% of available CPUs.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias')
sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias/study')
os.chdir('/home/gykovacs/workspaces/good/conditioning_bias/study')

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (
    KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit, StratifiedShuffleSplit,
)
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import t as t_dist

from conditioning_bias import apply as cb_apply

RANDOM_SEED = 5
N_JOBS = max(1, int(0.8 * os.cpu_count()))

MAX_DEPTH_GRID = [None] + list(range(2, 16))
TRAIN_SIZE = 0.9  # repeated holdout, replaces the K-fold sweep
HOLDOUT_K_EQUIV = 10  # 1/(k-1) = 1/9 = 0.1/0.9 exactly; reuses the K-fold-derived corrected_p formula
N_REPEATS = 100  # corrected-test variance floor saturates well before this, see paper section 4.4
REF_N_SPLITS, REF_N_REPEATS = 5, 20  # reference selection, matches original 001-model-selection

EXCLUDED_REGR = ['o-ring', 'diabetes']
NEW_REGR = ['concrete', 'boom_bikes']

SKLEARN_CLASSES = {
    'dtc': DecisionTreeClassifier, 'rfc': RandomForestClassifier,
    'dtr': DecisionTreeRegressor, 'rfr': RandomForestRegressor,
}
IS_FOREST = {'dtc': False, 'rfc': True, 'dtr': False, 'rfr': True}
AVG_KIND = {'dtc': 'full', 'rfc': 'half', 'dtr': 'full', 'rfr': 'half'}
IS_CLASSIF = {'dtc': True, 'rfc': True, 'dtr': False, 'rfr': False}


# ---------------------------------------------------------------------------
# core prediction helpers (truncated traversal, no refitting per depth candidate)
# ---------------------------------------------------------------------------
def _leaf_values(tree, X, operator, max_depth=None):
    node_ids = cb_apply(X, tree, operator=operator, max_depth=max_depth)
    return tree.tree_.value[node_ids, 0, :]


def _predict_single(tree, X, operator, is_classif, max_depth=None):
    vals = _leaf_values(tree, X, operator, max_depth)
    if is_classif:
        probs = (vals.T / np.sum(vals, axis=1)).T
        return probs[:, 1]
    return vals[:, 0]


def _predict_forest(trees, X, operator, is_classif, max_depth=None):
    preds = [_predict_single(t, X, operator, is_classif, max_depth) for t in trees]
    return np.mean(preds, axis=0)


def _predict_avg(model_or_trees, X, is_classif, is_forest, avg_kind, max_depth=None):
    if not is_forest:
        p_leq = _predict_single(model_or_trees, X, '<=', is_classif, max_depth)
        p_lt = _predict_single(model_or_trees, X, '<', is_classif, max_depth)
        return (p_leq + p_lt) / 2
    trees = model_or_trees
    if avg_kind == 'full':
        p_leq = _predict_forest(trees, X, '<=', is_classif, max_depth)
        p_lt = _predict_forest(trees, X, '<', is_classif, max_depth)
        return (p_leq + p_lt) / 2
    n_half = len(trees) // 2
    p1 = _predict_forest(trees[:n_half], X, '<=', is_classif, max_depth)
    p2 = _predict_forest(trees[n_half:], X, '<', is_classif, max_depth)
    return (p1 + p2) / 2


def _score(y_true, y_pred, is_classif):
    return roc_auc_score(y_true, y_pred) if is_classif else r2_score(y_true, y_pred)


def _fit_unconstrained(cls, X, y, is_forest):
    m = cls(max_depth=None, min_samples_leaf=1, random_state=RANDOM_SEED).fit(X, y)
    return m.estimators_ if is_forest else m


def _predict_any(fit, X, operator, is_classif, is_forest, max_depth=None):
    return (_predict_forest(fit, X, operator, is_classif, max_depth) if is_forest
            else _predict_single(fit, X, operator, is_classif, max_depth))


def _count_lattice(tree, X):
    """Nodes whose threshold coincides with an observed value of the split feature, / total
    internal (non-leaf) nodes, for a single fitted sklearn tree."""
    leaf = tree.tree_.children_left == tree.tree_.children_right
    n_lattice, n_splits = 0, 0
    for node in np.where(~leaf)[0]:
        f = tree.tree_.feature[node]
        t = tree.tree_.threshold[node]
        n_splits += 1
        if np.any(np.isclose(X[:, f], t)):
            n_lattice += 1
    return n_lattice, n_splits


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


# ---------------------------------------------------------------------------
# stage 1: reference hyperparameter selection (descriptive; also yields rho/rho_k)
# ---------------------------------------------------------------------------
def reference_selection(estimator_key, X, y):
    cls = SKLEARN_CLASSES[estimator_key]
    is_forest = IS_FOREST[estimator_key]
    is_classif = IS_CLASSIF[estimator_key]

    if is_classif:
        n_minority = int(min(np.sum(y == c) for c in np.unique(y)))
        k_eff = min(REF_N_SPLITS, n_minority) if n_minority >= 2 else 2
        validator = RepeatedStratifiedKFold(n_splits=k_eff, n_repeats=REF_N_REPEATS, random_state=RANDOM_SEED)
        splits = list(validator.split(X, y, y))
    else:
        validator = RepeatedKFold(n_splits=REF_N_SPLITS, n_repeats=REF_N_REPEATS, random_state=RANDOM_SEED)
        splits = list(validator.split(X))

    scores_by_depth = {d: [] for d in MAX_DEPTH_GRID}
    n_lat_total, n_split_total = 0, 0
    for train, test in splits:
        fit = _fit_unconstrained(cls, X[train], y[train], is_forest)
        trees = fit if is_forest else [fit]
        for t in trees:
            a, b = _count_lattice(t, X[train])
            n_lat_total += a
            n_split_total += b
        for d in MAX_DEPTH_GRID:
            pred = _predict_any(fit, X[test], '<=', is_classif, is_forest, d)
            scores_by_depth[d].append(_score(y[test], pred, is_classif))

    mean_scores = {d: np.mean(v) for d, v in scores_by_depth.items()}
    best_depth = max(mean_scores, key=mean_scores.get)
    rho_k = n_lat_total / n_split_total if n_split_total else 0.0

    # rho: fit to ALL data with the selected depth
    full_fit = _fit_unconstrained(cls, X, y, is_forest)
    trees = full_fit if is_forest else [full_fit]
    n_lat_full, n_split_full = 0, 0
    for t in trees:
        a, b = _count_lattice(t, X)
        n_lat_full += a
        n_split_full += b
    rho = n_lat_full / n_split_full if n_split_full else 0.0

    return {
        'best_depth': best_depth, 'best_score': mean_scores[best_depth],
        'rho': rho, 'rho_k': rho_k,
    }


# ---------------------------------------------------------------------------
# stage 2: properly nested outer fold (hyperparameter AND operator selection, no leakage)
# ---------------------------------------------------------------------------
def _outer_fold(estimator_key, X, y, train_idx, test_idx):
    cls = SKLEARN_CLASSES[estimator_key]
    is_forest = IS_FOREST[estimator_key]
    is_classif = IS_CLASSIF[estimator_key]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    if is_classif:
        n_minority = int(min(np.sum(y_train == c) for c in np.unique(y_train)))
        inner_k = min(5, n_minority) if n_minority >= 2 else 2
        inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=RANDOM_SEED)
        inner_splits = list(inner_cv.split(X_train, y_train))
    else:
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        inner_splits = list(inner_cv.split(X_train))

    inner_fits = [_fit_unconstrained(cls, X_train[itr], y_train[itr], is_forest)
                  for itr, ival in inner_splits]

    depth_scores = {d: [] for d in MAX_DEPTH_GRID}
    for (itr, ival), fit in zip(inner_splits, inner_fits):
        for d in MAX_DEPTH_GRID:
            pred = _predict_any(fit, X_train[ival], '<=', is_classif, is_forest, d)
            depth_scores[d].append(_score(y_train[ival], pred, is_classif))
    mean_depth_scores = {d: np.mean(v) for d, v in depth_scores.items()}
    best_depth = max(mean_depth_scores, key=mean_depth_scores.get)

    op_scores = {'<=': [], '<': []}
    for (itr, ival), fit in zip(inner_splits, inner_fits):
        for op in ['<=', '<']:
            pred = _predict_any(fit, X_train[ival], op, is_classif, is_forest, best_depth)
            op_scores[op].append(_score(y_train[ival], pred, is_classif))
    selected_op = '<=' if np.mean(op_scores['<=']) >= np.mean(op_scores['<']) else '<'

    final_fit = cls(max_depth=best_depth, min_samples_leaf=1, random_state=RANDOM_SEED).fit(X_train, y_train)
    final = final_fit.estimators_ if is_forest else final_fit

    leq = _predict_any(final, X_test, '<=', is_classif, is_forest)
    lt = _predict_any(final, X_test, '<', is_classif, is_forest)
    avg = _predict_avg(final, X_test, is_classif, is_forest, AVG_KIND[estimator_key])
    select = leq if selected_op == '<=' else lt

    return {
        'leq': _score(y_test, leq, is_classif), 'lt': _score(y_test, lt, is_classif),
        'avg': _score(y_test, avg, is_classif), 'select': _score(y_test, select, is_classif),
    }


def run_nested_holdout(estimator_key, name, X, y):
    """
    Repeated random holdout (ShuffleSplit) at TRAIN_SIZE=0.9, replacing the K-fold sweep.
    n_test/n_train = 0.1/0.9 = 1/9, which is exactly the K=10 K-fold floor (1/(k-1)=1/9), so the
    existing corrected_p(diffs, k) formula is reused unchanged with k=HOLDOUT_K_EQUIV -- this is
    not an approximation, the two ratios are identical by construction, just realized via
    independent random splits (Nadeau & Bengio 2003's original setting) instead of a non-
    overlapping K-fold partition (Bouckaert & Frank 2004's adaptation of the same correction).
    """
    is_classif = IS_CLASSIF[estimator_key]

    if is_classif:
        validator = StratifiedShuffleSplit(n_splits=N_REPEATS, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)
        splits = list(validator.split(X, y))
    else:
        validator = ShuffleSplit(n_splits=N_REPEATS, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)
        splits = list(validator.split(X))

    out = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(_outer_fold)(estimator_key, X, y, tr, te) for tr, te in splits
    )
    df = pd.DataFrame(out)
    rows = []

    leq, lt, avg, select = df['leq'].values, df['lt'].values, df['avg'].values, df['select'].values
    worst = np.minimum(leq, lt)
    non_tied = leq != lt

    pairs = {
        '<=_vs_<': leq - lt,
        'avg_vs_worst': avg - worst,
        'select_vs_avg': select - avg,
        'select_vs_worst': select - worst,
    }
    for pair, diffs in pairs.items():
        p, m = corrected_p(diffs, HOLDOUT_K_EQUIV)
        rows.append({'estimator': estimator_key, 'name': name, 'pair': pair,
                     'mean_diff': m, 'corrected_p': p, 'n_valid': len(diffs),
                     'mean_leq': leq.mean(), 'mean_lt': lt.mean(),
                     'mean_avg': avg.mean(), 'mean_select': select.mean()})

    # tie between leq/lt already excluded (non_tied); does the selected operator's value
    # match whichever of leq/lt is actually better on this held-out split?
    better_is_leq = leq[non_tied] > lt[non_tied]
    select_is_leq = np.isclose(select[non_tied], leq[non_tied])
    hit = (select_is_leq == better_is_leq).astype(float)
    n_valid = int(non_tied.sum())
    if n_valid >= 2:
        p, m = corrected_p(hit - 0.5, HOLDOUT_K_EQUIV)
        m = m + 0.5
    else:
        p, m = np.nan, np.nan
    rows.append({'estimator': estimator_key, 'name': name, 'pair': 'select_hit_rate',
                 'mean_diff': m, 'corrected_p': p, 'n_valid': n_valid,
                 'mean_leq': leq.mean(), 'mean_lt': lt.mean(),
                 'mean_avg': avg.mean(), 'mean_select': select.mean()})
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def get_regr_pool():
    import common_datasets.regression as regr
    from datasets import regr_datasets
    continuing = regr_datasets[~regr_datasets['name'].isin(EXCLUDED_REGR)][
        ['name', 'data_loader_function']]
    all_loaders = regr.get_filtered_data_loaders(n_bounds=(1, 1_000_000), n_from_phenotypes=1)
    summary = regr.get_summary_pdf()
    summary = summary[summary['data_loader_function'].isin(all_loaders)]
    new_rows = summary[summary['name'].isin(NEW_REGR)][['name', 'data_loader_function']]
    return pd.concat([continuing, new_rows], ignore_index=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()

    from datasets import binclas_datasets
    print(f"N_JOBS = {N_JOBS} (80% of {os.cpu_count()} cores)", flush=True)

    if args.benchmark:
        from datasets import regr_datasets
        N_REPEATS = 20
        for est, name, table in [
            ('dtc', 'appendicitis', binclas_datasets), ('rfc', 'appendicitis', binclas_datasets),
            ('dtr', 'wsn-ale', regr_datasets), ('rfr', 'wsn-ale', regr_datasets),
        ]:
            row = table[table['name'] == name].iloc[0]
            data = row['data_loader_function']()
            X, y = data['data'], data['target']
            t0 = time.time()
            ref = reference_selection(est, X, y)
            t_ref = time.time() - t0

            t0 = time.time()
            rows = run_nested_holdout(est, name, X, y)
            t_sweep = time.time() - t0

            print(f"{est}/{name}: ref selection {t_ref:.1f}s (depth={ref['best_depth']}, "
                  f"rho={ref['rho']:.3f}); holdout train_size=0.9,n_repeats={N_REPEATS} "
                  f"{t_sweep:.1f}s ({t_sweep/N_REPEATS:.3f}s/split)")
        sys.exit(0)

    ref_rows = []
    holdout_rows = []

    def process(estimator_key, name, X, y):
        t0 = time.time()
        ref = reference_selection(estimator_key, X, y)
        ref.update({'estimator': estimator_key, 'name': name})
        ref_rows.append(ref)
        rows = run_nested_holdout(estimator_key, name, X, y)
        holdout_rows.extend(rows)
        print(f"{estimator_key} / {name}: ref_depth={ref['best_depth']} "
              f"rho={ref['rho']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(ref_rows).to_csv('data/params_maxdepth_only.csv', index=False)
        pd.DataFrame(holdout_rows).to_csv('data/holdout_results_maxdepth_only.csv', index=False)

    for _, row in binclas_datasets.iterrows():
        name = row['name']
        data = row['data_loader_function']()
        X, y = data['data'], data['target']
        for est in ['dtc', 'rfc']:
            process(est, name, X, y)

    for _, row in get_regr_pool().iterrows():
        name = row['name']
        data = row['data_loader_function']()
        X, y = data['data'], data['target']
        for est in ['dtr', 'rfr']:
            process(est, name, X, y)

    print("DONE", flush=True)
