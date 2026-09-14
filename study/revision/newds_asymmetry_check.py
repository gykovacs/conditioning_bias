"""
Same stability spot-check that caught o-ring's instability (model_selection_lt.py), applied to
just the 2 new replacement datasets (concrete, boom_bikes) -- confirms they don't quietly
reproduce the same problem before trusting their K-sweep numbers.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias')
sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias/study')
os.chdir('/home/gykovacs/workspaces/good/conditioning_bias/study')

import common_datasets.regression as regr
from conditioning_bias import OperatorDecisionTreeRegressor, OperatorRandomForestRegressor
from evaluation import evaluate_regression
from config import data_dir, random_seed, n_splits_ms, n_repeats_ms

NEW_NAMES = ['concrete', 'boom_bikes']
all_loaders = regr.get_filtered_data_loaders(n_bounds=(1, 1_000_000), n_from_phenotypes=1)
summary = regr.get_summary_pdf()
summary = summary[summary['data_loader_function'].isin(all_loaders)].copy()
new_datasets = summary[summary['name'].isin(NEW_NAMES)][
    ['name', 'citation_key', 'n_col', 'n', 'data_loader_function']
].reset_index(drop=True)

max_depth = [None] + list(range(2, 16))
min_samples_leaf = np.round(np.logspace(np.log10(0.005), np.log10(0.2), 15), 3)

configurations = [
    {'estimator': OperatorDecisionTreeRegressor, 'label': 'dtr'},
    {'estimator': OperatorRandomForestRegressor, 'label': 'rfr'},
]

for conf in configurations:
    label = conf['label']
    results_md = evaluate_regression(
        datasets=new_datasets, estimator=conf['estimator'],
        params=[{'random_state': random_seed, 'mode': '<=', 'max_depth': md} for md in max_depth],
        validator_params={'n_splits': n_splits_ms, 'n_repeats': n_repeats_ms, 'random_state': random_seed},
        random_seed=random_seed, modes=['<'],
    )
    results_msl = evaluate_regression(
        datasets=new_datasets, estimator=conf['estimator'],
        params=[{'random_state': random_seed, 'mode': '<=', 'min_samples_leaf': msl} for msl in min_samples_leaf],
        validator_params={'n_splits': n_splits_ms, 'n_repeats': n_repeats_ms, 'random_state': random_seed},
        random_seed=random_seed, modes=['<'],
    )
    data = pd.concat([results_md, results_msl])
    data['params_str'] = data['params'].astype(str)
    grouped = data.groupby(['name', 'estimator', 'params_str']).agg({'r2': 'mean'}).reset_index()
    grouped['params'] = grouped['params_str']
    best_lt = grouped.groupby('name', group_keys=False).apply(lambda p: p.sort_values('r2').iloc[-1]).reset_index(drop=True)

    leq = pd.read_csv(os.path.join(data_dir, f'params_{label}.csv'))
    leq = leq[leq['name'].isin(NEW_NAMES)]
    merged = pd.merge(leq, best_lt[['name', 'params', 'r2']], on='name', suffixes=('_leq', '_lt'))
    def parse(s):
        d = eval(s, {'np': np}); d.pop('random_state', None); d.pop('mode', None); return d
    merged['parsed_leq'] = merged['params_leq'].apply(parse)
    merged['parsed_lt'] = merged['params_lt'].apply(parse)
    merged['differ'] = merged.apply(lambda r: r['parsed_leq'] != r['parsed_lt'], axis=1)
    merged['score_diff'] = (merged['r2_leq'] - merged['r2_lt']).abs()
    print(f"\n{label}:")
    print(merged[['name', 'parsed_leq', 'parsed_lt', 'differ', 'r2_leq', 'r2_lt', 'score_diff']].to_string(index=False))
