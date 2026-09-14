"""
Generates tab_biasedinstance_v5.tex (reviewer 3, point 13: performance restricted to instances
that actually hit a biased node) from study/data/biased_instance_results.csv, and prints the
summary statistics quoted in the paper text.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest, combine_pvalues

DATA_DIR = '/home/gykovacs/workspaces/good/conditioning_bias/study/data'
OUT_DIR = '/home/gykovacs/workspaces/good/conditioning_bias/paper_2023_conditioning_bias_springer'
MAIN_HOLDOUT = f'{DATA_DIR}/holdout_results_maxdepth_only.csv'

df = pd.read_csv(f'{DATA_DIR}/biased_instance_results.csv')
main = pd.read_csv(MAIN_HOLDOUT)
main_aw = main[main.pair == 'avg_vs_worst'][['estimator', 'name', 'mean_diff']].copy()
main_aw['abs_diff'] = main_aw['mean_diff'].abs()
merged = df.merge(main_aw, on=['estimator', 'name'])

EST_LABEL = {'dtc': 'DT classif.', 'rfc': 'RF classif.', 'dtr': 'DT regr.', 'rfr': 'RF regr.'}


def fmt_sci(v):
    if v is None or pd.isna(v):
        return '--'
    s = f'{v:.1e}'
    if s in ('0.0e+00', '-0.0e+00'):
        return '0'
    if s == '1.0e+00':
        return '1'
    return s

lines = []
for est in ['dtc', 'rfc', 'dtr', 'rfr']:
    sub = df[df.estimator == est]
    msub = merged[merged.estimator == est]
    hit_median = sub['hit_rate'].median()
    corr = msub['hit_rate'].corr(msub['abs_diff'], method='spearman')

    qualified = sub[sub.n_valid_splits >= 5].copy()
    n_qual = len(qualified)
    if n_qual >= 2:
        n_pos = int((qualified.avgworst_meandiff > 0).sum())
        n_neg = int((qualified.avgworst_meandiff < 0).sum())
        n_used = n_pos + n_neg
        sign_p = binomtest(n_pos, n_used, 0.5).pvalue if n_used > 0 else np.nan
        p_clipped = qualified['avgworst_p'].clip(lower=1e-300).dropna().values
        fisher_p = combine_pvalues(p_clipped, method='fisher')[1] if len(p_clipped) else np.nan
        nsig = int((qualified['avgworst_p'] < 0.05).sum())
        wl = f"{n_pos}/{n_neg}"
        sign_s = fmt_sci(sign_p)
        fisher_s = fmt_sci(fisher_p)
        nsig_s = str(nsig)
    else:
        wl = "--"; sign_s = "--"; fisher_s = "--"; nsig_s = "--"

    lines.append(f"{EST_LABEL[est]} & {hit_median*100:.2f}\\% & {corr:.2f} & {n_qual}/20 & "
                 f"{wl} & {sign_s} & {fisher_s} & {nsig_s} \\\\")

header = (r"\begin{tabular}{l@{\hspace{4pt}}r@{\hspace{4pt}}r@{\hspace{4pt}}r@{\hspace{4pt}}"
          r"r@{\hspace{4pt}}r@{\hspace{4pt}}r@{\hspace{4pt}}r}" + "\n\\toprule\n"
          r"estimator & hit rate & $r_S$(hit,\ eff.) & $n$ qual. & wins/losses & "
          r"sign $p$ & Fisher $p$ & $n_{sig}$ \\" + "\n"
          r"\midrule")
footer = r"\bottomrule" + "\n" + r"\end{tabular}"
with open(f'{OUT_DIR}/tab_biasedinstance_v5.tex', 'w') as f:
    f.write(header + "\n" + "\n".join(lines) + "\n" + footer + "\n")

print("table written")

# ---- summary stats for the paper text ----
print()
for est in ['dtc', 'rfc', 'dtr', 'rfr']:
    sub = df[df.estimator == est]
    print(f"{est}: hit_rate median={sub.hit_rate.median():.4f} min={sub.hit_rate.min():.4f} "
          f"max={sub.hit_rate.max():.4f}")

print()
for est in ['rfc', 'rfr']:
    sub = df[df.estimator == est]
    worst_full = np.minimum(sub.full_leq, sub.full_lt)
    hit_valid = sub.dropna(subset=['hit_leq'])
    worst_hit = np.minimum(hit_valid.hit_leq, hit_valid.hit_lt)
    full_benefit = (sub.full_avg - worst_full).mean()
    hit_benefit = (hit_valid.hit_avg - worst_hit).mean()
    print(f"{est}: full-set avg-worst benefit={full_benefit:.5f}  "
          f"hit-subset avg-worst benefit={hit_benefit:.5f}  ratio={hit_benefit/full_benefit:.1f}x")
