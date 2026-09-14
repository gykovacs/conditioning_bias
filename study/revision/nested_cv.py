"""
Properly nested CV for BOTH hyperparameter selection and operator selection, parallelized across
outer folds at 80% of available CPUs, with the max_depth dimension of the hyperparameter grid
evaluated by truncating one full-depth fit instead of refitting per candidate.

Why truncation is valid here even though it doesn't reproduce sklearn's own refit bit-for-bit:
whenever a candidate max_depth refit and a truncated full-depth tree disagree, it's because some
node's best split was tied with an alternative, and sklearn's random tie-break (whose outcome
depends on how much of the shared RNG stream prior tree-growth consumed) picked differently in
the two cases. A tie means both splits are equally optimal under CART's own training objective --
the truncated tree isn't a worse max_depth=d model, it's a different, equally-valid one. (The
random RNG-consumption argument only fires at genuine ties -- if a node's best split is unique,
no tie-break is invoked and truncation reproduces the refit exactly regardless of depth.)

This does NOT extend to min_samples_leaf: that constraint removes candidate splits from
consideration entirely (not just caps depth), so a split forbidden under a larger min_samples_leaf
isn't "tied" with whatever real fit would pick instead -- the real fit explores a materially
different, non-tied alternative. So min_samples_leaf candidates are still real, separate fits.

Per outer fold:
  1. For each of 5 inner folds: fit ONE tree/forest per estimator with max_depth=None,
     min_samples_leaf=1 (fully unconstrained). Evaluate all 14 max_depth candidates via truncated
     traversal (conditioning_bias.apply/tree_inference's max_depth parameter) -- no refitting.
     Separately, fit each of the 15 min_samples_leaf candidates for real (max_depth=None).
     All scored with the default operator ('<='), matching the original (non-nested) methodology.
  2. Pick the best of the 29 candidates by mean inner-CV score -> best hyperparameter setting.
  3. Reuse whichever fit(s) correspond to that setting (no extra fits) to compare '<=' vs '<' on
     the same inner-validation folds -> selected operator.
  4. Refit once on the full outer-training fold with the selected hyperparameter, evaluate '<=',
     '<', 'avg', and 'select' (the nested-selected operator) on the outer-test fold.
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
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, r2_score

from conditioning_bias import apply as cb_apply

RANDOM_SEED = 5
N_JOBS = max(1, int(0.8 * os.cpu_count()))

MAX_DEPTH_GRID = [None] + list(range(2, 16))
MIN_SAMPLES_LEAF_GRID = list(np.round(np.logspace(np.log10(0.005), np.log10(0.2), 15), 3))

SKLEARN_CLASSES = {
    'dtc': DecisionTreeClassifier, 'rfc': RandomForestClassifier,
    'dtr': DecisionTreeRegressor, 'rfr': RandomForestRegressor,
}
IS_FOREST = {'dtc': False, 'rfc': True, 'dtr': False, 'rfr': True}
AVG_KIND = {'dtc': 'full', 'rfc': 'half', 'dtr': 'full', 'rfr': 'half'}


def _leaf_values(tree, X, operator, max_depth=None):
    """Predicted value distribution (classification: per-class counts; regression: [mean]) for a
    single fitted sklearn tree, via the operator-aware, optionally depth-truncated traversal."""
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


def _fit(cls, params, X, y, random_state, is_forest):
    kwargs = dict(params)
    kwargs['random_state'] = random_state
    m = cls(**kwargs).fit(X, y)
    return m.estimators_ if is_forest else m


def _outer_fold(estimator_key, X, y, train_idx, test_idx, is_classif):
    cls = SKLEARN_CLASSES[estimator_key]
    is_forest = IS_FOREST[estimator_key]
    avg_kind = AVG_KIND[estimator_key]

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

    # one unconstrained fit per inner fold, covers every max_depth candidate via truncation
    unconstrained_fits = []
    for itr, ival in inner_splits:
        m = _fit(cls, {'max_depth': None, 'min_samples_leaf': 1}, X_train[itr], y_train[itr],
                 RANDOM_SEED, is_forest)
        unconstrained_fits.append(m)

    candidate_scores = {}  # ('max_depth', d) or ('min_samples_leaf', v) -> mean inner score
    for d in MAX_DEPTH_GRID:
        scores = []
        for (itr, ival), fit in zip(inner_splits, unconstrained_fits):
            pred = (_predict_forest(fit, X_train[ival], '<=', is_classif, d) if is_forest
                    else _predict_single(fit, X_train[ival], '<=', is_classif, d))
            scores.append(_score(y_train[ival], pred, is_classif))
        candidate_scores[('max_depth', d)] = np.mean(scores)

    msl_fits = {}  # min_samples_leaf value -> list of per-inner-fold fits (real, separate)
    for msl in MIN_SAMPLES_LEAF_GRID:
        fits = []
        scores = []
        for itr, ival in inner_splits:
            m = _fit(cls, {'max_depth': None, 'min_samples_leaf': msl}, X_train[itr], y_train[itr],
                      RANDOM_SEED, is_forest)
            fits.append(m)
            pred = (_predict_forest(m, X_train[ival], '<=', is_classif) if is_forest
                    else _predict_single(m, X_train[ival], '<=', is_classif))
            scores.append(_score(y_train[ival], pred, is_classif))
        candidate_scores[('min_samples_leaf', msl)] = np.mean(scores)
        msl_fits[msl] = fits

    best_key = max(candidate_scores, key=candidate_scores.get)
    best_kind, best_val = best_key

    # operator selection: reuse the fits already made for the winning hyperparameter, no refit
    op_scores = {'<=': [], '<': []}
    if best_kind == 'max_depth':
        for (itr, ival), fit in zip(inner_splits, unconstrained_fits):
            for op in ['<=', '<']:
                pred = (_predict_forest(fit, X_train[ival], op, is_classif, best_val) if is_forest
                        else _predict_single(fit, X_train[ival], op, is_classif, best_val))
                op_scores[op].append(_score(y_train[ival], pred, is_classif))
    else:
        for (itr, ival), fit in zip(inner_splits, msl_fits[best_val]):
            for op in ['<=', '<']:
                pred = (_predict_forest(fit, X_train[ival], op, is_classif) if is_forest
                        else _predict_single(fit, X_train[ival], op, is_classif))
                op_scores[op].append(_score(y_train[ival], pred, is_classif))
    selected_op = '<=' if np.mean(op_scores['<=']) >= np.mean(op_scores['<']) else '<'

    # single refit on the full outer-training fold with the selected hyperparameter
    best_params = {'max_depth': best_val, 'min_samples_leaf': 1} if best_kind == 'max_depth' \
        else {'max_depth': None, 'min_samples_leaf': best_val}
    final_fit = _fit(cls, best_params, X_train, y_train, RANDOM_SEED, is_forest)

    results = {}
    for mode, label in [('<=', 'leq'), ('<', 'lt')]:
        pred = (_predict_forest(final_fit, X_test, mode, is_classif) if is_forest
                else _predict_single(final_fit, X_test, mode, is_classif))
        results[label] = _score(y_test, pred, is_classif)
    avg_pred = _predict_avg(final_fit, X_test, is_classif, is_forest, avg_kind)
    results['avg'] = _score(y_test, avg_pred, is_classif)
    sel_pred = (_predict_forest(final_fit, X_test, selected_op, is_classif) if is_forest
                else _predict_single(final_fit, X_test, selected_op, is_classif))
    results['select'] = _score(y_test, sel_pred, is_classif)
    results['selected_hyperparam'] = f'{best_kind}={best_val}'
    results['selected_operator'] = selected_op
    return results


def run_nested(estimator_key, name, X, y, is_classif, k=10, n_repeats=10):
    if is_classif:
        n_minority = int(min(np.sum(y == c) for c in np.unique(y)))
        k_eff = min(k, n_minority) if n_minority >= 2 else 2
        validator = RepeatedStratifiedKFold(n_splits=k_eff, n_repeats=n_repeats, random_state=RANDOM_SEED)
        splits = list(validator.split(X, y, y))
    else:
        validator = RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=RANDOM_SEED)
        splits = list(validator.split(X))

    t0 = time.time()
    out = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(_outer_fold)(estimator_key, X, y, tr, te, is_classif) for tr, te in splits
    )
    elapsed = time.time() - t0
    df = pd.DataFrame(out)
    df['estimator'] = estimator_key
    df['name'] = name
    df['fold'] = range(len(df))
    return df, elapsed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()

    from datasets import binclas_datasets, regr_datasets
    print(f"N_JOBS = {N_JOBS} (80% of {os.cpu_count()} cores)")

    if args.benchmark:
        for est, name, is_classif, k, n_repeats in [
            ('dtc', 'appendicitis', True, 10, 2),
            ('rfc', 'appendicitis', True, 10, 2),
            ('dtr', 'wsn-ale', False, 10, 2),
            ('rfr', 'wsn-ale', False, 10, 2),
        ]:
            table = binclas_datasets if is_classif else regr_datasets
            row = table[table['name'] == name].iloc[0]
            data = row['data_loader_function']()
            X, y = data['data'], data['target']
            df, elapsed = run_nested(est, name, X, y, is_classif, k=k, n_repeats=n_repeats)
            n_folds = len(df)
            print(f"{est}/{name}: {n_folds} outer folds in {elapsed:.1f}s "
                  f"({elapsed/n_folds:.2f}s/fold) -- mean leq={df['leq'].mean():.4f} "
                  f"lt={df['lt'].mean():.4f} avg={df['avg'].mean():.4f} select={df['select'].mean():.4f}")
