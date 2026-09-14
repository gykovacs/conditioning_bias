"""
Answers two questions from the user:
  1. Would MORE repeats (n_repeats > 400) push borderline cells over the significance line?
  2. How close are cells to significance, and would a LARGER K (fold count) help more?

Mathematical background: the Bouckaert & Frank corrected variance is
    corrected_var = var_d * (1/n + 1/(k-1))
where n = n_repeats * k. As n_repeats -> infinity, 1/n -> 0, so corrected_var approaches a FLOOR
of var_d / (k-1) that does NOT shrink with more repeats -- it only shrinks by increasing k.
This script demonstrates that numerically on the existing data (subsampling repeats), and
estimates (assuming var_d stays roughly similar, which needs a real re-run to confirm) what
larger k would do to the closest near-miss cells.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

DATA_DIR = '/home/gykovacs/workspaces/good/conditioning_bias/study/data'
K = 5
N_REPEATS = 400

ESTIMATORS = {
    'dtc': ('auc', ['<=', '<', 'avg_full']),
    'dtr': ('r2',  ['<=', '<', 'avg_full']),
    'rfc': ('auc', ['<=', '<', 'avg_half']),
    'rfr': ('r2',  ['<=', '<', 'avg_half']),
}


def corrected_p(diffs, k):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean_d = np.mean(diffs)
    var_d = np.var(diffs, ddof=1)
    if var_d == 0:
        return 1.0
    corrected_var = var_d * (1.0 / n + 1.0 / (k - 1))
    tstat = mean_d / np.sqrt(corrected_var)
    df = n - 1
    return 2 * (1 - t_dist.cdf(np.abs(tstat), df)), var_d, mean_d


def load_diffs(estimator_key, name, a_mode, b_mode):
    score, _ = ESTIMATORS[estimator_key]
    df = pd.read_csv(os.path.join(DATA_DIR, f'evaluation_{estimator_key}.csv'))
    df = df[(df['name'] == name) & (df['mode'].isin([a_mode, b_mode]))].sort_values('fold')
    a = df[df['mode'] == a_mode].sort_values('fold')[score].values
    b = df[df['mode'] == b_mode].sort_values('fold')[score].values
    return a - b  # in repeat-major fold order, length N_REPEATS*K


cases = [
    ('rfr', 'plastic', '<=', '<'),
    ('dtr', 'plastic', '<=', '<'),
    ('rfr', 'excitation_current', '<', 'avg_half'),
    ('rfc', 'bupa', '<=', '<'),
    ('rfr', 'wankara', '<=', '<'),
]

print("=" * 90)
print("Q1: does increasing n_repeats (subsample of what we already have, up to 400) help?")
print("=" * 90)
for est, name, a, b in cases:
    diffs = load_diffs(est, name, a, b)
    print(f"\n{est} / {name} / {a} vs {b}  (full n_repeats=400 corrected p = "
          f"{corrected_p(diffs, K)[0]:.4f})")
    for n_rep in [25, 50, 100, 200, 400]:
        sub = diffs[:n_rep * K]
        p, var_d, mean_d = corrected_p(sub, K)
        n = len(sub)
        floor = var_d / (K - 1)
        n_term = var_d / n
        print(f"  n_repeats={n_rep:4d} (n={n:5d}): corrected p={p:.4f}   "
              f"[1/n term={n_term:.2e} vs floor var_d/(k-1)={floor:.2e}  "
              f"-> floor dominates by {floor/max(n_term,1e-30):.0f}x]")

print()
print("=" * 90)
print("Q2: holding var_d fixed at the observed (k=5) value, what would larger K project to?")
print("(APPROXIMATION ONLY -- a real re-run at k=10/20 is needed since var_d itself may change)")
print("=" * 90)
for est, name, a, b in cases:
    diffs = load_diffs(est, name, a, b)
    var_d = np.var(diffs, ddof=1)
    mean_d = np.mean(diffs)
    n = len(diffs)
    print(f"\n{est} / {name} / {a} vs {b}  (observed mean_diff={mean_d:.2e}, var_d={var_d:.2e})")
    for k_hyp in [5, 10, 15, 20, 30]:
        corrected_var = var_d * (1.0 / n + 1.0 / (k_hyp - 1))
        tstat = mean_d / np.sqrt(corrected_var)
        p = 2 * (1 - t_dist.cdf(np.abs(tstat), n - 1))
        print(f"  k={k_hyp:2d}: projected corrected p={p:.4f}   (floor var_d/(k-1)={var_d/(k_hyp-1):.2e})")
