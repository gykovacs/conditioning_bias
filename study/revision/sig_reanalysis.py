"""
Re-analysis of the significance testing methodology used in study/007 and study/008.

Compares four ways of testing (default vs non-default vs averaged) on the SAME already-computed
per-fold results in study/data/evaluation_{dtc,dtr,rfc,rfr}.csv:

  1. naive: current paper's method -- paired Wilcoxon signed-rank on all 2000 (=400 repeats x 5
     folds) per-fold scores, treated as independent pairs.
  2. repeat_mean: paired Wilcoxon on 400 per-repeat MEAN scores (mean over the 5 folds of each
     repeat) -- reduces (but does not fully remove) the fold-overlap non-independence.
  3. corrected_t: Bouckaert & Frank (2004) / Nadeau & Bengio corrected resampled paired t-test,
     applied to the 2000 raw per-fold differences, correcting the variance estimate for the known
     train/test overlap ratio of k-fold CV (correction factor 1/(k-1) for k=5).
  4. unpaired: reviewer 2's suggested unpaired (Mann-Whitney/ranksums) test treating the 2000
     scores of each mode as two independent unpaired samples (ignores the pairing entirely).

For every (estimator x dataset x mode-pair) triple, records the significance call (alpha=0.05)
under each method, so we can see how much of the current paper's positive results survive.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ranksums, t as t_dist

DATA_DIR = '/home/gykovacs/workspaces/good/conditioning_bias/study/data'
K = 5          # folds per repeat
N_REPEATS = 400

ESTIMATORS = {
    'dtc': ('auc', ['<=', '<', 'avg_full']),
    'dtr': ('r2',  ['<=', '<', 'avg_full']),
    'rfc': ('auc', ['<=', '<', 'avg_half']),
    'rfr': ('r2',  ['<=', '<', 'avg_half']),
}

ALPHA = 0.05


def corrected_resampled_ttest(diffs, k=K):
    """Bouckaert & Frank (2004) corrected resampled paired t-test for repeated k-fold CV.
    diffs: array of n = n_repeats * k paired differences (A - B), in repeat-major fold order.
    Correction factor accounts for the 1/(k-1) test/train size ratio of k-fold CV.
    Returns two-sided p-value.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean_d = np.mean(diffs)
    var_d = np.var(diffs, ddof=1)
    if var_d == 0:
        return 1.0 if mean_d == 0 else 0.0
    corrected_var = var_d * (1.0 / n + 1.0 / (k - 1))
    tstat = mean_d / np.sqrt(corrected_var)
    df = n - 1
    p = 2 * (1 - t_dist.cdf(np.abs(tstat), df))
    return p


def paired_wilcoxon_min(a, b):
    """Exactly the paper's current convention: min of the two one-sided p-values, NOT doubled."""
    p_less = wilcoxon(a, b, alternative='less', zero_method='zsplit').pvalue
    p_gr = wilcoxon(a, b, alternative='greater', zero_method='zsplit').pvalue
    return min(p_less, p_gr)


def analyze(estimator_key):
    score, modes = ESTIMATORS[estimator_key]
    df = pd.read_csv(os.path.join(DATA_DIR, f'evaluation_{estimator_key}.csv'))
    df = df[df['mode'].isin(modes)]

    df['repeat'] = df['fold'] // K
    df['fold_in_repeat'] = df['fold'] % K

    rows = []
    for name, gdf in df.groupby('name'):
        pivot_raw = {}
        pivot_repeat_mean = {}
        for mode in modes:
            sub = gdf[gdf['mode'] == mode].sort_values('fold')
            assert len(sub) == N_REPEATS * K, (estimator_key, name, mode, len(sub))
            pivot_raw[mode] = sub[score].values
            pivot_repeat_mean[mode] = sub.groupby('repeat')[score].mean().values

        pairs = [('<=', '<'), ('<=', modes[2]), ('<', modes[2])]
        for a_mode, b_mode in pairs:
            a_raw, b_raw = pivot_raw[a_mode], pivot_raw[b_mode]
            a_rm, b_rm = pivot_repeat_mean[a_mode], pivot_repeat_mean[b_mode]

            p_naive = paired_wilcoxon_min(a_raw, b_raw)
            p_repeat_mean = paired_wilcoxon_min(a_rm, b_rm)
            p_corrected = corrected_resampled_ttest(a_raw - b_raw, k=K)
            p_unpaired = ranksums(a_raw, b_raw).pvalue

            rows.append({
                'estimator': estimator_key,
                'name': name,
                'pair': f'{a_mode}_vs_{b_mode}',
                'mean_a': np.mean(a_raw),
                'mean_b': np.mean(b_raw),
                'mean_diff': np.mean(a_raw) - np.mean(b_raw),
                'p_naive': p_naive,
                'p_repeat_mean': p_repeat_mean,
                'p_corrected_t': p_corrected,
                'p_unpaired': p_unpaired,
                'sig_naive': p_naive < ALPHA,
                'sig_repeat_mean': p_repeat_mean < ALPHA,
                'sig_corrected_t': p_corrected < ALPHA,
                'sig_unpaired': p_unpaired < ALPHA,
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    all_results = pd.concat([analyze(k) for k in ESTIMATORS], ignore_index=True)
    out_path = '/tmp/claude-1000/-home-gykovacs-workspaces-good-conditioning-bias/6e8bc03a-b501-4c6f-93a3-979405538152/scratchpad/sig_reanalysis_results.csv'
    all_results.to_csv(out_path, index=False)

    print(f"Total (estimator x dataset x pair) rows: {len(all_results)}")
    print()
    for method in ['sig_naive', 'sig_repeat_mean', 'sig_corrected_t', 'sig_unpaired']:
        print(f"{method:20s}: {all_results[method].sum():4d} / {len(all_results)} significant "
              f"({100*all_results[method].mean():.1f}%)")

    print()
    print("Cross-tab: naive-significant cells, how do the alternative methods call them?")
    naive_sig = all_results[all_results['sig_naive']]
    print(f"  naive significant: {len(naive_sig)}")
    for method in ['sig_repeat_mean', 'sig_corrected_t', 'sig_unpaired']:
        agree = naive_sig[method].sum()
        print(f"    also significant under {method}: {agree} / {len(naive_sig)} "
              f"({100*agree/len(naive_sig):.1f}%)")

    print()
    print("Breakdown by estimator and pair:")
    summary = all_results.groupby(['estimator', 'pair'])[
        ['sig_naive', 'sig_repeat_mean', 'sig_corrected_t', 'sig_unpaired']
    ].sum()
    print(summary)
