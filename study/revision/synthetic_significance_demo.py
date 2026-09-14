"""
Core promise of the synthetic-data approach (validation plan section 2a): sidestep the whole
repeated-CV non-independence problem entirely, by drawing genuinely fresh i.i.d. data for every
replicate. No correction needed -- an ordinary paired t-test on independent replicates is valid
by construction.

Protocol per replicate:
  1. draw a fresh training set (fixed size) from the generative process
  2. fit the tree/forest with a fixed hyperparameter (chosen once, not tuned per replicate --
     avoids reintroducing the §1b hyperparameter-selection leakage issue here)
  3. draw a large, fresh, independent i.i.d. test set (n_test_eval) and score each mode on it
  4. record the score difference between modes for this replicate

Repeat for n_replicates independent replicates -> n_replicates independent (not 2000
fold-correlated) paired differences -> valid ordinary paired t-test / Wilcoxon.

Also runs a small sensitivity sweep over discrete-domain granularity and tree depth, since that's
directly relevant to reviewer 1's "why does bagging not average it out" and the general "how big
is the effect really" questions.
"""
import sys
import numpy as np
from scipy.stats import wilcoxon, ttest_1samp

sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias')
from synthetic import make_conditioning_bias_regression, make_conditioning_bias_classification
from conditioning_bias import (
    OperatorDecisionTreeRegressor, OperatorRandomForestRegressor,
    OperatorDecisionTreeClassifier, OperatorRandomForestClassifier,
)
from sklearn.metrics import r2_score, roc_auc_score

N_TRAIN = 500
N_TEST_EVAL = 200_000
N_REPLICATES = 200


def run_regression_replicates(estimator_cls, est_kwargs, domain_size, depth, seed0=0):
    diffs_leq_lt = []
    diffs_leq_avg = []
    for rep in range(N_REPLICATES):
        seed = seed0 * 100_000 + rep
        X_train, y_train = make_conditioning_bias_regression(
            N_TRAIN, discrete_domain_size=domain_size, random_state=seed)
        X_test, y_test = make_conditioning_bias_regression(
            N_TEST_EVAL, discrete_domain_size=domain_size, random_state=seed + 500_000)

        model = estimator_cls(mode='<=', random_state=seed, max_depth=depth, **est_kwargs)
        model.fit(X_train, y_train)

        model.set_mode('<=')
        r2_leq = r2_score(y_test, model.predict(X_test))
        model.set_mode('<')
        r2_lt = r2_score(y_test, model.predict(X_test))
        avg_mode = 'avg_full' if 'Forest' not in estimator_cls.__name__ else 'avg_half'
        model.set_mode(avg_mode)
        r2_avg = r2_score(y_test, model.predict(X_test))

        diffs_leq_lt.append(r2_leq - r2_lt)
        diffs_leq_avg.append(r2_avg - min(r2_leq, r2_lt))

    return np.array(diffs_leq_lt), np.array(diffs_leq_avg)


def summarize(name, diffs):
    diffs = np.array(diffs)
    mean_d = diffs.mean()
    t_p = ttest_1samp(diffs, 0.0).pvalue
    w_p = wilcoxon(diffs, zero_method='zsplit').pvalue if np.any(diffs != 0) else 1.0
    print(f"  {name}: mean_diff={mean_d:.5f}  std={diffs.std():.5f}  "
          f"paired-t p={t_p:.2e}  Wilcoxon p={w_p:.2e}  {'***' if t_p < 0.05 else ''}")


if __name__ == '__main__':
    print("=== Sensitivity sweep: DecisionTreeRegressor, varying domain granularity ===")
    for domain_size in [5, 11, 21, 51]:
        print(f"\ndiscrete_domain_size={domain_size}, max_depth=8, n_train={N_TRAIN}, "
              f"n_test_eval={N_TEST_EVAL}, n_replicates={N_REPLICATES}")
        diffs_lt, diffs_avg = run_regression_replicates(
            OperatorDecisionTreeRegressor, {}, domain_size, depth=8, seed0=domain_size)
        summarize("<= vs <", diffs_lt)
        summarize("avg vs worst(<=,<)", diffs_avg)

    print("\n=== Sensitivity sweep: RandomForestRegressor, varying domain granularity ===")
    for domain_size in [5, 11, 21]:
        print(f"\ndiscrete_domain_size={domain_size}, max_depth=8, n_train={N_TRAIN}, "
              f"n_test_eval={N_TEST_EVAL}, n_replicates={N_REPLICATES}")
        diffs_lt, diffs_avg = run_regression_replicates(
            OperatorRandomForestRegressor, {'n_estimators': 50}, domain_size, depth=8,
            seed0=domain_size + 1000)
        summarize("<= vs <", diffs_lt)
        summarize("avg vs worst(<=,<)", diffs_avg)

    print("\n=== Sensitivity sweep: DecisionTreeRegressor, varying max_depth (domain_size=11) ===")
    for depth in [3, 5, 8, 12]:
        print(f"\nmax_depth={depth}, discrete_domain_size=11, n_train={N_TRAIN}")
        diffs_lt, diffs_avg = run_regression_replicates(
            OperatorDecisionTreeRegressor, {}, 11, depth=depth, seed0=depth + 2000)
        summarize("<= vs <", diffs_lt)
        summarize("avg vs worst(<=,<)", diffs_avg)
