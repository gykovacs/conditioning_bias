"""
Pareto-chart upgrade of the heavy-tail figure: rank-ordered |mean_diff| (avg vs worst(<=,<)) bars
(log scale, left axis) plus a cumulative-share-of-total-effect line (right axis, 0-100%), one panel
per estimator. Makes the concentration of the effect directly readable (e.g., "the top 3 datasets
already account for X% of the total effect"), rather than only visible from the bar heights.
Also prints the Gini coefficient of the effect-size distribution per estimator, for the paper text.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

df = pd.read_csv('/home/gykovacs/workspaces/good/conditioning_bias/study/data/holdout_results_maxdepth_only.csv')

titles = {
    'dtc': 'Decision tree, classification (AUC)',
    'rfc': 'Random forest, classification (AUC)',
    'dtr': 'Decision tree, regression ($r^2$)',
    'rfr': 'Random forest, regression ($r^2$)',
}


def gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
axes = axes.flatten()

for ax, est in zip(axes, ['dtc', 'rfc', 'dtr', 'rfr']):
    sub = df[(df.estimator == est) & (df.pair == 'avg_vs_worst')].copy()
    sub['abs_diff'] = sub['mean_diff'].abs().clip(lower=1e-12)
    sub = sub.sort_values('abs_diff', ascending=False).reset_index(drop=True)
    sig = sub['corrected_p'] < 0.05
    ranks = np.arange(1, len(sub) + 1)

    colors = ['#c0392b' if s else '#95a5a6' for s in sig]
    ax.bar(ranks, sub['abs_diff'], color=colors, width=0.7, zorder=2)
    ax.set_yscale('log')
    ax.set_title(titles[est], fontsize=10)
    ax.set_xlabel('dataset rank (by effect size)', fontsize=9)
    ax.set_ylabel(r'$|$mean score difference$|$', fontsize=9)
    ax.set_xticks(range(1, len(sub) + 1, 2))
    ax.tick_params(labelsize=8)

    cum_share = 100 * np.cumsum(sub['abs_diff'].values) / sub['abs_diff'].values.sum()
    ax2 = ax.twinx()
    ax2.plot(ranks, cum_share, color='#1f4e8c', marker='o', markersize=3, linewidth=1.5, zorder=3)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel('cumulative share of total effect (%)', fontsize=8, color='#1f4e8c')
    ax2.tick_params(axis='y', labelsize=8, labelcolor='#1f4e8c')
    ax2.axhline(80, color='#1f4e8c', linestyle=':', linewidth=0.7, alpha=0.5)

    g = gini(sub['abs_diff'].values)
    ax.text(0.97, 0.95, f'Gini = {g:.2f}', transform=ax.transAxes, ha='right', va='top',
             fontsize=8, bbox=dict(boxstyle='round', fc='white', ec='#888', alpha=0.85))

handles = [plt.Rectangle((0, 0), 1, 1, color='#c0392b'), plt.Rectangle((0, 0), 1, 1, color='#95a5a6'),
           plt.Line2D([0], [0], color='#1f4e8c', marker='o', markersize=4, linewidth=1.5)]
fig.legend(handles, ['individually significant ($p<0.05$)', 'not significant', 'cumulative share of total effect'],
           loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.035, 1, 1])
out = '/home/gykovacs/workspaces/good/conditioning_bias/paper_2023_conditioning_bias_springer/heavytail_v6.pdf'
plt.savefig(out, bbox_inches='tight')
print('saved', out)

print()
for est in ['dtc', 'rfc', 'dtr', 'rfr']:
    sub = df[(df.estimator == est) & (df.pair == 'avg_vs_worst')].copy()
    vals = sub['mean_diff'].abs().sort_values(ascending=False).values
    g = gini(vals)
    cum_share = np.cumsum(vals) / vals.sum()
    n_for_80 = int(np.searchsorted(cum_share, 0.80) + 1)
    print(f'{est}: gini={g:.3f}  top3_share={cum_share[2]:.2f}  top5_share={cum_share[4]:.2f}  '
          f'n_datasets_for_80pct={n_for_80}/20')
