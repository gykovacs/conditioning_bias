"""
Sets up the two replacement regression datasets (concrete, boom_bikes -- replacing o-ring and
diabetes, the two datasets whose small size made 5-fold-CV hyperparameter selection unstable).
Mirrors study/001-model-selection-{md,msl}.ipynb, 001-optimal-parameters.ipynb, and
study/005-evaluation.ipynb exactly, but only for these 2 datasets, so everything else already
computed for the other 18 regression datasets (and all 20 classification datasets, which are
unaffected) can be reused as-is.
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
from config import data_dir, random_seed, n_splits_ms, n_repeats_ms, n_splits, n_repeats

NEW_NAMES = ['concrete', 'boom_bikes']

all_loaders = regr.get_filtered_data_loaders(n_bounds=(1, 1_000_000), n_from_phenotypes=1)
summary = regr.get_summary_pdf()
summary = summary[summary['data_loader_function'].isin(all_loaders)].copy()
new_datasets = summary[summary['name'].isin(NEW_NAMES)][
    ['name', 'citation_key', 'n_col', 'n', 'data_loader_function']
].reset_index(drop=True)
print(new_datasets[['name', 'n', 'n_col']], flush=True)

max_depth = [None] + list(range(2, 16))
min_samples_leaf = np.round(np.logspace(np.log10(0.005), np.log10(0.2), 15), 3)

configurations = [
    {'estimator': OperatorDecisionTreeRegressor, 'label': 'dtr'},
    {'estimator': OperatorRandomForestRegressor, 'label': 'rfr'},
]

print("=== model selection for new datasets ===", flush=True)
for conf in configurations:
    label = conf['label']
    results_md = evaluate_regression(
        datasets=new_datasets, estimator=conf['estimator'],
        params=[{'random_state': random_seed, 'mode': '<=', 'max_depth': md} for md in max_depth],
        validator_params={'n_splits': n_splits_ms, 'n_repeats': n_repeats_ms, 'random_state': random_seed},
        random_seed=random_seed, modes=['<='],
    )
    results_md.to_csv(os.path.join(data_dir, f'model_selection_{label}_md_newds.csv'))

    results_msl = evaluate_regression(
        datasets=new_datasets, estimator=conf['estimator'],
        params=[{'random_state': random_seed, 'mode': '<=', 'min_samples_leaf': msl} for msl in min_samples_leaf],
        validator_params={'n_splits': n_splits_ms, 'n_repeats': n_repeats_ms, 'random_state': random_seed},
        random_seed=random_seed, modes=['<='],
    )
    results_msl.to_csv(os.path.join(data_dir, f'model_selection_{label}_msl_newds.csv'))
    print(f"{label}: model selection sweeps done", flush=True)

print("=== selecting optimal params for new datasets ===", flush=True)
new_params = {}
for conf in configurations:
    label = conf['label']
    data = pd.concat([
        pd.read_csv(os.path.join(data_dir, f'model_selection_{label}_md_newds.csv')),
        pd.read_csv(os.path.join(data_dir, f'model_selection_{label}_msl_newds.csv')),
    ])
    grouped = data.groupby(['name', 'estimator', 'params']).agg({'r2': 'mean'}).reset_index()
    best_params = grouped.groupby('name', group_keys=False).apply(
        lambda p: p.sort_values('r2').iloc[-1]).reset_index(drop=True)
    best_params = best_params[['name', 'params', 'r2']]
    new_params[label] = best_params
    print(best_params, flush=True)

    # merge into the existing params_{label}.csv (append, since these are new dataset names)
    existing = pd.read_csv(os.path.join(data_dir, f'params_{label}.csv'))
    existing = existing[~existing['name'].isin(NEW_NAMES)]
    combined = pd.concat([existing, best_params], ignore_index=True)
    combined.to_csv(os.path.join(data_dir, f'params_{label}.csv'), index=False)

print("=== baseline K=5, n_repeats=400 evaluation for new datasets ===", flush=True)
for conf in configurations:
    label = conf['label']
    params_lookup = {row['name']: eval(row['params'], {'np': np}) for _, row in new_params[label].iterrows()}
    results = evaluate_regression(
        datasets=new_datasets, estimator=conf['estimator'],
        params=params_lookup,
        validator_params={'n_splits': n_splits, 'n_repeats': n_repeats, 'random_state': random_seed},
        random_seed=random_seed,
        modes=['<=', '<', 'avg_full' if label == 'dtr' else 'avg_half'],
    )
    results.to_csv(os.path.join(data_dir, f'evaluation_{label}_newds.csv'))
    print(f"{label}: baseline evaluation done", flush=True)

print("DONE", flush=True)
