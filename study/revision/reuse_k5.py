"""
Extracts K=5 results for the meta-analysis by reanalyzing ALREADY-COMPUTED data:
  - all 20 classification datasets: study/data/evaluation_{dtc,rfc}.csv (unchanged)
  - 18 continuing regression datasets: study/data/evaluation_{dtr,rfr}.csv, filtered to exclude
    o-ring/diabetes
  - 2 new regression datasets (concrete, boom_bikes): study/data/evaluation_{dtr,rfr}_newds.csv
    (from new_datasets_setup.py -- must have completed before this runs)

No refitting, no new experiments -- just applies the same corrected resampled t-test used
throughout to data that already exists, in the same long format as real_data_ksweep.py's output
(k=5 rows), so the two can be concatenated directly for the meta-analysis.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

DATA_DIR = '/home/gykovacs/workspaces/good/conditioning_bias/study/data'
K = 5
EXCLUDED_REGR = ['o-ring', 'diabetes']

ESTIMATORS = {
    'dtc': ('auc', ['<=', '<', 'avg_full']),
    'rfc': ('auc', ['<=', '<', 'avg_half']),
    'dtr': ('r2', ['<=', '<', 'avg_full']),
    'rfr': ('r2', ['<=', '<', 'avg_half']),
}


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


def process(estimator_key, exclude_names=()):
    base_key = estimator_key.replace('_newds', '')
    score, modes = ESTIMATORS[base_key]
    fname = f'evaluation_{estimator_key}.csv'
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df = df[df['mode'].isin(modes)]
    if exclude_names:
        df = df[~df['name'].isin(exclude_names)]

    rows = []
    for name, gdf in df.groupby('name'):
        pivot = {}
        for mode in modes:
            sub = gdf[gdf['mode'] == mode].sort_values('fold')
            pivot[mode] = sub[score].values
        leq, lt, avg = pivot[modes[0]], pivot[modes[1]], pivot[modes[2]]
        diffs_lt = leq - lt
        diffs_avg = avg - np.minimum(leq, lt)

        p1, m1 = corrected_p(diffs_lt, K)
        p2, m2 = corrected_p(diffs_avg, K)
        rows.append({'estimator': base_key, 'name': name, 'k': K, 'pair': '<=_vs_<',
                     'mean_diff': m1, 'corrected_p': p1})
        rows.append({'estimator': base_key, 'name': name, 'k': K, 'pair': 'avg_vs_worst',
                     'mean_diff': m2, 'corrected_p': p2})
    return rows


if __name__ == '__main__':
    all_rows = []
    for key in ['dtc', 'rfc']:
        all_rows.extend(process(key))
    for key in ['dtr', 'rfr']:
        all_rows.extend(process(key, exclude_names=EXCLUDED_REGR))
    for key in ['dtr_newds', 'rfr_newds']:
        all_rows.extend(process(key))

    out = pd.DataFrame(all_rows)
    out.to_csv(os.path.join(DATA_DIR, 'ksweep_results_k5_reused.csv'), index=False)
    print(f"{len(out)} rows written")
    print(out.groupby('estimator')['name'].nunique())
