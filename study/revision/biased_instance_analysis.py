"""
Reviewer 3, point 13: "do a final analysis to show the performance of the models for only the
instances that actually hit at least one biased node with the threshold value. Also the
percentage of instances that hit these nodes for each dataset would be of interest."

This reuses full_experiment.py's exact nested per-split procedure (same seeds, same splits, same
hyperparameter+operator selection) so the numbers here are directly comparable to the paper's main
results -- we are not introducing a second, separately-tuned pipeline. The only addition is: for
each outer-split test instance, does its <=-traversal path (equivalently the <-traversal path,
since they coincide up to the first tie) pass through any node whose threshold exactly equals the
instance's value on the split feature? For a forest, an instance counts as "hit" if this happens
in at least one tree.

Because the hit subset of any single outer split's test portion (10% of a dataset) can be very
small, per-split scores restricted to the hit subset are not reliable (an AUC on 2-3 records is not
meaningful). We therefore report two things per (estimator, dataset):
  1. hit_rate: the fraction of test instances, pooled across all R=100 splits, that hit a biased
     node -- a large-sample, well-estimated quantity.
  2. pooled scores on the hit subset (and, for contrast, on the full test set) computed by pooling
     predictions across all splits and scoring once -- a descriptive statistic, not a significance
     test (pooling non-independent splits for inference is exactly the anti-conservative mistake
     the rest of this study corrects for; pooling for a single descriptive point estimate has no
     such problem, since no p-value is attached to it).
We additionally attempt the paper's own corrected paired t-test on the hit-subset score
differences, restricted to splits with enough hit instances for the score to be well-defined,
reporting how many splits qualified -- honestly showing where this breaks down for small datasets.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias')
sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias/study')
sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias/study/revision')
os.chdir('/home/gykovacs/workspaces/good/conditioning_bias/study')

import full_experiment as fe
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

MIN_HIT_FOR_SCORE = 5  # minimum hit instances (and, for classification, both classes present)
                        # required to compute a per-split score restricted to the hit subset


def _hits_biased_node(tree, X, tol=1e-9):
    """
    Per-instance boolean: does the (default, <=) traversal path of X through `tree` pass through
    any internal node whose threshold exactly (within tol) equals the instance's value on the
    split feature? This is well-defined regardless of operator, since the <= and < paths coincide
    up to the first such tie by construction (see conditioning_bias/_tree_inference.py apply()).
    """
    node_ids = np.zeros(X.shape[0], dtype=int)
    hit = np.zeros(X.shape[0], dtype=bool)
    leaf_flag = tree.tree_.children_left == tree.tree_.children_right

    while True:
        active = np.where(~leaf_flag[node_ids])[0]
        if len(active) == 0:
            break
        nodes = node_ids[active]
        feature = tree.tree_.feature[nodes]
        threshold = tree.tree_.threshold[nodes]
        vals = X[active, feature]
        tied = np.isclose(vals, threshold, atol=tol, rtol=0)
        hit[active[tied]] = True
        left = vals <= threshold
        nodes[left] = tree.tree_.children_left[nodes[left]]
        nodes[~left] = tree.tree_.children_right[nodes[~left]]
        node_ids[active] = nodes
    return hit


def _forest_hit(trees, X):
    hits = np.array([_hits_biased_node(t, X) for t in trees])  # (n_trees, n_instances)
    return hits.any(axis=0), hits.mean(axis=0)


def _outer_fold_biased(estimator_key, X, y, train_idx, test_idx):
    """Identical selection procedure to full_experiment._outer_fold, plus hit-mask instrumentation
    on the outer test portion using the final refit model."""
    cls = fe.SKLEARN_CLASSES[estimator_key]
    is_forest = fe.IS_FOREST[estimator_key]
    is_classif = fe.IS_CLASSIF[estimator_key]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    if is_classif:
        n_minority = int(min(np.sum(y_train == c) for c in np.unique(y_train)))
        inner_k = min(5, n_minority) if n_minority >= 2 else 2
        inner_cv = fe.StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=fe.RANDOM_SEED)
        inner_splits = list(inner_cv.split(X_train, y_train))
    else:
        inner_cv = fe.KFold(n_splits=5, shuffle=True, random_state=fe.RANDOM_SEED)
        inner_splits = list(inner_cv.split(X_train))

    inner_fits = [fe._fit_unconstrained(cls, X_train[itr], y_train[itr], is_forest)
                  for itr, ival in inner_splits]

    depth_scores = {d: [] for d in fe.MAX_DEPTH_GRID}
    for (itr, ival), fit in zip(inner_splits, inner_fits):
        for d in fe.MAX_DEPTH_GRID:
            pred = fe._predict_any(fit, X_train[ival], '<=', is_classif, is_forest, d)
            depth_scores[d].append(fe._score(y_train[ival], pred, is_classif))
    mean_depth_scores = {d: np.mean(v) for d, v in depth_scores.items()}
    best_depth = max(mean_depth_scores, key=mean_depth_scores.get)

    op_scores = {'<=': [], '<': []}
    for (itr, ival), fit in zip(inner_splits, inner_fits):
        for op in ['<=', '<']:
            pred = fe._predict_any(fit, X_train[ival], op, is_classif, is_forest, best_depth)
            op_scores[op].append(fe._score(y_train[ival], pred, is_classif))
    selected_op = '<=' if np.mean(op_scores['<=']) >= np.mean(op_scores['<']) else '<'

    final_fit = cls(max_depth=best_depth, min_samples_leaf=1, random_state=fe.RANDOM_SEED).fit(X_train, y_train)
    final = final_fit.estimators_ if is_forest else final_fit

    leq = fe._predict_any(final, X_test, '<=', is_classif, is_forest)
    lt = fe._predict_any(final, X_test, '<', is_classif, is_forest)
    avg = fe._predict_avg(final, X_test, is_classif, is_forest, fe.AVG_KIND[estimator_key])
    select = leq if selected_op == '<=' else lt

    if is_forest:
        hit, frac_trees_hit = _forest_hit(final, X_test)
    else:
        hit = _hits_biased_node(final, X_test)
        frac_trees_hit = hit.astype(float)

    return {
        'y': y_test, 'leq': leq, 'lt': lt, 'avg': avg, 'select': select,
        'hit': hit, 'frac_trees_hit': frac_trees_hit,
    }


def run_dataset(estimator_key, name, X, y):
    is_classif = fe.IS_CLASSIF[estimator_key]
    if is_classif:
        validator = StratifiedShuffleSplit(n_splits=fe.N_REPEATS, train_size=fe.TRAIN_SIZE,
                                            random_state=fe.RANDOM_SEED)
        splits = list(validator.split(X, y))
    else:
        validator = ShuffleSplit(n_splits=fe.N_REPEATS, train_size=fe.TRAIN_SIZE,
                                  random_state=fe.RANDOM_SEED)
        splits = list(validator.split(X))

    out = Parallel(n_jobs=fe.N_JOBS, backend='loky')(
        delayed(_outer_fold_biased)(estimator_key, X, y, tr, te) for tr, te in splits
    )

    # --- sanity check: on the NOT-hit subset, leq and lt predictions must be bit-identical ---
    for i, r in enumerate(out):
        not_hit = ~r['hit']
        if not np.allclose(r['leq'][not_hit], r['lt'][not_hit]):
            raise AssertionError(f"{estimator_key}/{name} split {i}: leq != lt on a non-hit instance")

    # --- pooled descriptive statistics across all R splits ---
    hit_rate = np.mean(np.concatenate([r['hit'] for r in out]))
    n_total = sum(len(r['y']) for r in out)
    n_hit = int(np.sum(np.concatenate([r['hit'] for r in out])))
    frac_trees_hit_mean = np.mean(np.concatenate([r['frac_trees_hit'] for r in out]))

    def pooled_score(mask_fn):
        ys, leqs, lts, avgs, sels = [], [], [], [], []
        for r in out:
            m = mask_fn(r)
            if m.sum() == 0:
                continue
            ys.append(r['y'][m]); leqs.append(r['leq'][m]); lts.append(r['lt'][m])
            avgs.append(r['avg'][m]); sels.append(r['select'][m])
        if len(ys) == 0:
            return {'leq': np.nan, 'lt': np.nan, 'avg': np.nan, 'select': np.nan, 'n': 0}
        y_all = np.concatenate(ys)
        if is_classif and len(np.unique(y_all)) < 2:
            return {'leq': np.nan, 'lt': np.nan, 'avg': np.nan, 'select': np.nan, 'n': len(y_all)}
        return {
            'leq': fe._score(y_all, np.concatenate(leqs), is_classif),
            'lt': fe._score(y_all, np.concatenate(lts), is_classif),
            'avg': fe._score(y_all, np.concatenate(avgs), is_classif),
            'select': fe._score(y_all, np.concatenate(sels), is_classif),
            'n': len(y_all),
        }

    pooled_hit = pooled_score(lambda r: r['hit'])
    pooled_full = pooled_score(lambda r: np.ones_like(r['hit']))

    # --- best-effort corrected paired test, restricted to splits with enough hit instances ---
    def valid_for_score(r):
        m = r['hit']
        if m.sum() < MIN_HIT_FOR_SCORE:
            return False
        if is_classif and len(np.unique(r['y'][m])) < 2:
            return False
        return True

    valid_splits = [r for r in out if valid_for_score(r)]
    n_valid = len(valid_splits)
    test_results = {}
    for pair_name, a_key, b_key in [('leq_vs_lt', 'leq', 'lt'), ('avg_vs_worst', 'avg', None)]:
        if n_valid < 2:
            test_results[pair_name] = (np.nan, np.nan, n_valid)
            continue
        diffs = []
        for r in valid_splits:
            m = r['hit']
            s_a = fe._score(r['y'][m], r[a_key][m], is_classif)
            if b_key is None:
                s_leq = fe._score(r['y'][m], r['leq'][m], is_classif)
                s_lt = fe._score(r['y'][m], r['lt'][m], is_classif)
                s_b = min(s_leq, s_lt)
            else:
                s_b = fe._score(r['y'][m], r[b_key][m], is_classif)
            diffs.append(s_a - s_b)
        p, mean_d = fe.corrected_p(diffs, fe.HOLDOUT_K_EQUIV)
        test_results[pair_name] = (p, mean_d, n_valid)

    return {
        'estimator': estimator_key, 'name': name,
        'hit_rate': hit_rate, 'n_hit': n_hit, 'n_total': n_total,
        'frac_trees_hit_mean': frac_trees_hit_mean,
        'hit_leq': pooled_hit['leq'], 'hit_lt': pooled_hit['lt'],
        'hit_avg': pooled_hit['avg'], 'hit_select': pooled_hit['select'], 'hit_n': pooled_hit['n'],
        'full_leq': pooled_full['leq'], 'full_lt': pooled_full['lt'],
        'full_avg': pooled_full['avg'], 'full_select': pooled_full['select'], 'full_n': pooled_full['n'],
        'leqlt_p': test_results['leq_vs_lt'][0], 'leqlt_meandiff': test_results['leq_vs_lt'][1],
        'avgworst_p': test_results['avg_vs_worst'][0], 'avgworst_meandiff': test_results['avg_vs_worst'][1],
        'n_valid_splits': n_valid,
    }


if __name__ == '__main__':
    from datasets import binclas_datasets
    print(f"N_JOBS = {fe.N_JOBS}", flush=True)

    rows = []

    def process(estimator_key, name, X, y):
        t0 = time.time()
        row = run_dataset(estimator_key, name, X, y)
        rows.append(row)
        print(f"{estimator_key}/{name}: hit_rate={row['hit_rate']:.3f} "
              f"n_valid_splits={row['n_valid_splits']} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv('data/biased_instance_results.csv', index=False)

    for _, row in binclas_datasets.iterrows():
        name = row['name']
        data = row['data_loader_function']()
        X, y = data['data'], data['target']
        for est in ['dtc', 'rfc']:
            process(est, name, X, y)

    for _, row in fe.get_regr_pool().iterrows():
        name = row['name']
        data = row['data_loader_function']()
        X, y = data['data'], data['target']
        for est in ['dtr', 'rfr']:
            process(est, name, X, y)

    print("DONE", flush=True)
