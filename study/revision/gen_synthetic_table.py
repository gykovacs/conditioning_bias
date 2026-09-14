"""
Generates tab_synthetic_v5.tex from the standalone synthetic-validation experiment
(synthetic_experiment.py -> study/data/synthetic_results.csv), and prints summary statistics
(min/max Wilcoxon p for leq_vs_lt, min/max for avg_vs_worst) for use in the paper text.
"""
import pandas as pd
import numpy as np

DATA_DIR = '/home/gykovacs/workspaces/good/conditioning_bias/study/data'
OUT_DIR = '/home/gykovacs/workspaces/good/conditioning_bias/paper_2023_conditioning_bias_springer'

df = pd.read_csv(f'{DATA_DIR}/synthetic_results.csv')

EST_ORDER = ['dtc', 'rfc', 'dtr', 'rfr']
EST_LABEL = {'dtc': 'DT classif.', 'rfc': 'RF classif.', 'dtr': 'DT regr.', 'rfr': 'RF regr.'}


def fmt_score(v):
    return f'{v:.3f}'


def fmt_sci(v):
    s = f'{v:.1e}'
    if s in ('0.0e+00', '-0.0e+00'):
        return '0'
    if s == '1.0e+00':
        return '1'
    return s


def fmt_diff(v):
    return fmt_sci(v)


def mark(p):
    return r'$\ast$' if p < 0.05 else ''


PAIR_LABELS = {
    'leq_vs_lt': (r'$s_{\leq}$', r'$s_{<}$'),
    'avg_vs_worst': (r'$s_{avg}$', r'$s_{worst}$'),
}


def build_block(sweep, value_col, pair):
    sub = df[df.sweep == sweep]
    values = sorted(sub[value_col].unique())
    lines = []
    for v in values:
        cells = []
        for est in EST_ORDER:
            row = sub[(sub.estimator == est) & (sub[value_col] == v) & (sub.pair == pair)]
            if len(row) == 0:
                cells.append(' & & & & ')
                continue
            r = row.iloc[0]
            cells.append(f"{fmt_score(r['mean_a'])} & {fmt_score(r['mean_b'])} & "
                         f"{fmt_diff(r['mean_diff'])} & {fmt_sci(r['wilcoxon_p'])} & {mark(r['wilcoxon_p'])}")
        lines.append(f"{v} & " + ' & '.join(cells) + r' \\')
    return lines


one_block = r"r@{\hspace{5pt}}r@{\hspace{5pt}}r@{\hspace{5pt}}r@{\hspace{5pt}}c@{\hspace{10pt}}"
colspec = r"l@{\hspace{6pt}}" + one_block * 3 + r"r@{\hspace{5pt}}r@{\hspace{5pt}}r@{\hspace{5pt}}r@{\hspace{5pt}}c"
footer = r"\bottomrule" + "\n" + r"\end{tabular}"


def header_for(pair):
    a_label, b_label = PAIR_LABELS[pair]
    row2 = ' & '.join([f"{a_label} & {b_label} & diff & $p$ & "] * 4)
    return (
        r"\begin{tabular}{" + colspec + "}\n\\toprule\n"
        + r"& \multicolumn{5}{c}{DT classif.} & \multicolumn{5}{c}{RF classif.} & "
          r"\multicolumn{5}{c}{DT regr.} & \multicolumn{5}{c}{RF regr.} \\" + "\n"
        + f"value & {row2}" + r" \\" + "\n"
        + r"\midrule"
    )


def write_table(pair, out_name):
    domain_lines = build_block('domain', 'domain_size', pair)
    depth_lines = build_block('depth', 'depth', pair)
    with open(f'{OUT_DIR}/{out_name}.tex', 'w') as f:
        f.write(header_for(pair) + "\n")
        f.write(r"\multicolumn{21}{l}{\emph{Domain-size sweep (max depth fixed at 8)}} \\" + "\n")
        f.write("\n".join(domain_lines) + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\multicolumn{21}{l}{\emph{Depth sweep (domain size fixed at 11)}} \\" + "\n")
        f.write("\n".join(depth_lines) + "\n")
        f.write(footer + "\n")


write_table('avg_vs_worst', 'tab_synthetic_v5')
write_table('leq_vs_lt', 'tab_synthetic_null_v5')
print("synthetic tables written")

# summary stats for paper text
leq_lt = df[df.pair == 'leq_vs_lt']
avg_worst = df[df.pair == 'avg_vs_worst']

print(f"leq_vs_lt: n_configs={len(leq_lt)}, n_significant={int((leq_lt.wilcoxon_p < 0.05).sum())}, "
      f"p range [{leq_lt.wilcoxon_p.min():.2e}, {leq_lt.wilcoxon_p.max():.2e}], "
      f"|mean_diff| range [{leq_lt.mean_diff.abs().min():.2e}, {leq_lt.mean_diff.abs().max():.2e}]")

nonzero_effect = avg_worst[avg_worst.mean_diff.abs() > 1e-12]
zero_effect = avg_worst[avg_worst.mean_diff.abs() <= 1e-12]
print(f"avg_vs_worst: n_configs={len(avg_worst)}, n_significant={int((avg_worst.wilcoxon_p < 0.05).sum())}")
print(f"  configs with ~zero effect (n={len(zero_effect)}): "
      f"{list(zip(zero_effect.estimator, zero_effect.sweep, zero_effect.domain_size, zero_effect.depth))}")
if len(nonzero_effect):
    print(f"  configs with nonzero effect (n={len(nonzero_effect)}): "
          f"p range [{nonzero_effect.wilcoxon_p.min():.2e}, {nonzero_effect.wilcoxon_p.max():.2e}], "
          f"n_significant={int((nonzero_effect.wilcoxon_p < 0.05).sum())}")

for est in EST_ORDER:
    sub = avg_worst[avg_worst.estimator == est]
    print(f"  {est}: p range [{sub.wilcoxon_p.min():.2e}, {sub.wilcoxon_p.max():.2e}], "
          f"n_sig={int((sub.wilcoxon_p < 0.05).sum())}/{len(sub)}")
