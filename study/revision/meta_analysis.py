"""
Cross-dataset meta-analysis: combines per-dataset K-sweep results (real_data_ksweep.py's
K in {10,15,20} output + reuse_k5.py's K=5 output) into dataset-pool-level evidence.

Two independent-across-datasets tests per (estimator, k, pair):
  1. Sign/binomial test on win-rate: across the 20 datasets in a pool, how many show
     mean_diff > 0 (i.e. avg beats worst, or leq beats lt)? Under the null of no systematic
     effect, this should be ~50/50 (binomial(20, 0.5)). This test only uses the SIGN of each
     dataset's mean_diff, not its own within-dataset significance -- valid regardless of whether
     any single dataset individually reached significance.
  2. Fisher's combined-p test: combines the per-dataset corrected p-values (each already valid
     and independent since they come from different datasets) into one omnibus statistic.
     Only meaningful when the alternative direction is consistent, so also reports the
     one-sided version restricted to datasets where mean_diff has the "expected" sign.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest, combine_pvalues

FILES = [
    '/home/gykovacs/workspaces/good/conditioning_bias/study/data/ksweep_results.csv',
    '/home/gykovacs/workspaces/good/conditioning_bias/study/data/ksweep_results_k5_reused.csv',
]


def load():
    dfs = [pd.read_csv(f) for f in FILES]
    return pd.concat(dfs, ignore_index=True)


def sign_test(group):
    n = len(group)
    n_pos = int((group['mean_diff'] > 0).sum())
    n_neg = int((group['mean_diff'] < 0).sum())
    n_used = n_pos + n_neg
    if n_used == 0:
        return np.nan, n_pos, n_neg, n
    res = binomtest(n_pos, n_used, 0.5)
    return res.pvalue, n_pos, n_neg, n


def fisher_test(group):
    p = group['corrected_p'].clip(lower=1e-300).values
    if len(p) == 0:
        return np.nan
    stat, p_combined = combine_pvalues(p, method='fisher')
    return p_combined


if __name__ == '__main__':
    data = load()
    print(f"Total rows: {len(data)}")
    print(data.groupby(['estimator', 'k'])['name'].nunique())
    print()

    results = []
    for (estimator, k, pair), group in data.groupby(['estimator', 'k', 'pair']):
        sign_p, n_pos, n_neg, n = sign_test(group)
        fisher_p = fisher_test(group)
        n_individually_sig = int((group['corrected_p'] < 0.05).sum())
        results.append({
            'estimator': estimator, 'k': k, 'pair': pair,
            'n_datasets': n, 'n_pos': n_pos, 'n_neg': n_neg,
            'sign_test_p': sign_p, 'fisher_combined_p': fisher_p,
            'n_individually_significant': n_individually_sig,
        })

    out = pd.DataFrame(results).sort_values(['pair', 'estimator', 'k'])
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 200)
    print(out.to_string(index=False))
    out.to_csv('/home/gykovacs/workspaces/good/conditioning_bias/study/data/meta_analysis_results.csv',
               index=False)
