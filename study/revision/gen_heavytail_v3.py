"""
Heavy-tail figure v3: rank-ordered |mean_diff| (avg vs worst(<=,<)) across the 20-dataset pool,
one panel per estimator, from the redesigned (max_depth-only, repeated-holdout) methodology.
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

fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
axes = axes.flatten()

for ax, est in zip(axes, ['dtc', 'rfc', 'dtr', 'rfr']):
    sub = df[(df.estimator == est) & (df.pair == 'avg_vs_worst')].copy()
    sub['abs_diff'] = sub['mean_diff'].abs().clip(lower=1e-12)
    sub = sub.sort_values('abs_diff', ascending=False).reset_index(drop=True)
    sig = sub['corrected_p'] < 0.05

    colors = ['#c0392b' if s else '#95a5a6' for s in sig]
    ax.bar(np.arange(1, len(sub) + 1), sub['abs_diff'], color=colors, width=0.7)
    ax.set_yscale('log')
    ax.set_title(titles[est], fontsize=10)
    ax.set_xlabel('dataset rank (by effect size)', fontsize=9)
    ax.set_ylabel(r'$|$mean score difference$|$', fontsize=9)
    ax.set_xticks(range(1, len(sub) + 1, 2))
    ax.tick_params(labelsize=8)

handles = [plt.Rectangle((0, 0), 1, 1, color='#c0392b'), plt.Rectangle((0, 0), 1, 1, color='#95a5a6')]
fig.legend(handles, ['individually significant ($p<0.05$)', 'not significant'],
           loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.03, 1, 1])
out = '/home/gykovacs/workspaces/good/conditioning_bias/paper_2023_conditioning_bias_springer/heavytail_v3.pdf'
plt.savefig(out, bbox_inches='tight')
print('saved', out)

# print summary stats for the paper text
for est in ['dtc', 'rfc', 'dtr', 'rfr']:
    sub = df[(df.estimator == est) & (df.pair == 'avg_vs_worst')].copy()
    vals = sub['mean_diff'].abs().sort_values(ascending=False).values
    top3_share = vals[:3].sum() / vals.sum()
    print(f'{est}: max/median={vals.max()/np.median(vals):.1f}  top3_share={top3_share:.2f}  max={vals.max():.2e}')

    splits = pd.read_csv(f'/home/gykovacs/workspaces/good/conditioning_bias/study/data/params_maxdepth_only.csv')
    splits = splits[splits.estimator == est][['name', 'rho']]
    merged = sub.merge(splits, on='name')
    merged['abs_diff'] = merged['mean_diff'].abs()
    corr = merged['rho'].corr(merged['abs_diff'], method='spearman')
    print(f'  spearman(rho, abs_diff) = {corr:.3f}')
