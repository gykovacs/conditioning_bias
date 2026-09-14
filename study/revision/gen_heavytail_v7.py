"""
Two-panel version of the heavy-tail figure (v6 had one panel per estimator, four panels total).
Combines the two estimators of each problem type (decision tree + random forest) into a single
panel via grouped bars, so the figure keeps all four estimator/problem-type combinations but only
needs two panels: classification (dtc + rfc) and regression (dtr + rfr).

Per panel: rank-ordered |mean_diff| (avg vs worst(<=,<)) as grouped bars (log scale, left axis,
solid = decision tree, hatched = random forest, red = individually significant, gray = not), plus
one cumulative-share-of-total-effect line per estimator (right axis, 0-100%) and both estimators'
Gini coefficients annotated in the corner.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

df = pd.read_csv('/home/gykovacs/workspaces/good/conditioning_bias/study/data/holdout_results_maxdepth_only.csv')

PANELS = [
    ('Classification (AUC)', 'dtc', 'rfc'),
    ('Regression ($r^2$)', 'dtr', 'rfr'),
]
EST_LABEL = {'dtc': 'DT', 'rfc': 'RF', 'dtr': 'DT', 'rfr': 'RF'}
DT_COLOR = {'sig': '#c0392b', 'ns': '#95a5a6'}
LINE_COLOR = {'dtc': '#1f4e8c', 'rfc': '#e08214', 'dtr': '#1f4e8c', 'rfr': '#e08214'}


def gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def get_sorted(est):
    sub = df[(df.estimator == est) & (df.pair == 'avg_vs_worst')].copy()
    sub['abs_diff'] = sub['mean_diff'].abs().clip(lower=1e-12)
    return sub.sort_values('abs_diff', ascending=False).reset_index(drop=True)


fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

for ax, (title, est_dt, est_rf) in zip(axes, PANELS):
    sub_dt = get_sorted(est_dt)
    sub_rf = get_sorted(est_rf)
    ranks = np.arange(1, len(sub_dt) + 1)

    colors_dt = [DT_COLOR['sig'] if s else DT_COLOR['ns'] for s in sub_dt['corrected_p'] < 0.05]
    colors_rf = [DT_COLOR['sig'] if s else DT_COLOR['ns'] for s in sub_rf['corrected_p'] < 0.05]

    ax.bar(ranks - 0.2, sub_dt['abs_diff'], color=colors_dt, width=0.38, zorder=2,
           edgecolor='black', linewidth=0.3)
    ax.bar(ranks + 0.2, sub_rf['abs_diff'], color=colors_rf, width=0.38, zorder=2,
           hatch='//', edgecolor='black', linewidth=0.3)
    ax.set_yscale('log')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('dataset rank (by effect size)', fontsize=9)
    ax.set_ylabel(r'$|$mean score difference$|$', fontsize=9)
    ax.set_xticks(range(1, len(sub_dt) + 1, 2))
    ax.tick_params(labelsize=8)

    ax2 = ax.twinx()
    g_text = []
    for sub, est in [(sub_dt, est_dt), (sub_rf, est_rf)]:
        cum_share = 100 * np.cumsum(sub['abs_diff'].values) / sub['abs_diff'].values.sum()
        ax2.plot(ranks, cum_share, color=LINE_COLOR[est], marker='o', markersize=3,
                 linewidth=1.5, zorder=3)
        g_text.append(f"{EST_LABEL[est]} Gini = {gini(sub['abs_diff'].values):.2f}")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel('cumulative share of total effect (%)', fontsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.axhline(80, color='#555555', linestyle=':', linewidth=0.7, alpha=0.5)

    ax.text(0.97, 0.95, "\n".join(g_text), transform=ax.transAxes, ha='right', va='top',
            fontsize=8, bbox=dict(boxstyle='round', fc='white', ec='#888', alpha=0.85))

handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', linewidth=0.6),
    plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', hatch='//', linewidth=0.6),
    plt.Rectangle((0, 0), 1, 1, color=DT_COLOR['sig']),
    plt.Rectangle((0, 0), 1, 1, color=DT_COLOR['ns']),
    plt.Line2D([0], [0], color=LINE_COLOR['dtc'], marker='o', markersize=4, linewidth=1.5),
    plt.Line2D([0], [0], color=LINE_COLOR['rfc'], marker='o', markersize=4, linewidth=1.5),
]
labels = ['decision tree', 'random forest', 'individually significant ($p<0.05$)', 'not significant',
          'cumulative share (decision tree)', 'cumulative share (random forest)']
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.09))

plt.tight_layout(rect=[0, 0.1, 1, 1])
out = '/home/gykovacs/workspaces/good/conditioning_bias/paper_2023_conditioning_bias_springer/heavytail_v7.pdf'
plt.savefig(out, bbox_inches='tight')
print('saved', out)
