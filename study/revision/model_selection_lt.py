"""
Re-runs the EXACT model-selection procedure of study/001-model-selection-{md,msl}.ipynb and
study/001-optimal-parameters.ipynb, but scoring with the NON-DEFAULT operator '<' instead of the
paper's hardcoded '<='.

Purpose: test the hyperparameter-leakage/asymmetry hypothesis from the validation plan (section
1b) -- the current pipeline selects (max_depth / min_samples_leaf) to maximize performance with
'<=' only, then reuses those same hyperparameters when scoring '<'. If tuning explicitly for '<'
picks meaningfully different hyperparameters (or reaches a meaningfully different achieved score),
that's a structural, asymmetric advantage baked in favor of the default operator before the
conditioning-bias comparison even starts -- a plausible mechanical explanation for the
negative-skewed r_d^2 values reviewer 4 flagged, independent of any real property of '<' vs '<='.

Output: data/params_{label}_lt.csv (mirrors data/params_{label}.csv but selected for '<'), plus
a comparison printed to stdout.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias')
sys.path.insert(0, '/home/gykovacs/workspaces/good/conditioning_bias/study')
os.chdir('/home/gykovacs/workspaces/good/conditioning_bias/study')

from conditioning_bias import (OperatorDecisionTreeClassifier,
                                OperatorDecisionTreeRegressor,
                                OperatorRandomForestClassifier,
                                OperatorRandomForestRegressor)

from datasets import binclas_datasets, regr_datasets
from evaluation import evaluate_classification, evaluate_regression
from config import dataset_map, data_dir, random_seed, n_splits_ms, n_repeats_ms

max_depth = [None] + list(range(2, 16))
min_samples_leaf = np.round(np.logspace(np.log10(0.005), np.log10(0.2), 15), 3)

configurations = [
    {'datasets': binclas_datasets, 'estimator': OperatorDecisionTreeClassifier,
     'function': evaluate_classification, 'label': 'dtc'},
    {'datasets': binclas_datasets, 'estimator': OperatorRandomForestClassifier,
     'function': evaluate_classification, 'label': 'rfc'},
    {'datasets': regr_datasets, 'estimator': OperatorDecisionTreeRegressor,
     'function': evaluate_regression, 'label': 'dtr'},
    {'datasets': regr_datasets, 'estimator': OperatorRandomForestRegressor,
     'function': evaluate_regression, 'label': 'rfr'},
]

print("=== Running model selection sweeps with operator '<' (mirrors 001-model-selection-*) ===",
      flush=True)

for conf in configurations:
    label = conf['label']
    print(f"--- {label}: max_depth sweep ---", flush=True)
    results_md = conf['function'](
        datasets=conf['datasets'],
        estimator=conf['estimator'],
        params=[{'random_state': random_seed, 'mode': '<=', 'max_depth': md} for md in max_depth],
        validator_params={'n_splits': n_splits_ms, 'n_repeats': n_repeats_ms, 'random_state': random_seed},
        random_seed=random_seed,
        modes=['<'],
    )
    results_md['name'] = results_md['name'].apply(lambda x: dataset_map.get(x, x))
    results_md.to_csv(os.path.join(data_dir, f'model_selection_{label}_md_lt.csv'))

    print(f"--- {label}: min_samples_leaf sweep ---", flush=True)
    results_msl = conf['function'](
        datasets=conf['datasets'],
        estimator=conf['estimator'],
        params=[{'random_state': random_seed, 'mode': '<=', 'min_samples_leaf': msl} for msl in min_samples_leaf],
        validator_params={'n_splits': n_splits_ms, 'n_repeats': n_repeats_ms, 'random_state': random_seed},
        random_seed=random_seed,
        modes=['<'],
    )
    results_msl['name'] = results_msl['name'].apply(lambda x: dataset_map.get(x, x))
    results_msl.to_csv(os.path.join(data_dir, f'model_selection_{label}_msl_lt.csv'))

print("=== Selecting optimal params for '<' (mirrors 001-optimal-parameters) ===", flush=True)

inverse_map = {value: key for key, value in dataset_map.items()}
labels = ['dtc', 'dtr', 'rfc', 'rfr']

for label in labels:
    score = 'auc' if label[-1] == 'c' else 'r2'
    data = pd.concat([
        pd.read_csv(os.path.join(data_dir, f'model_selection_{label}_md_lt.csv')),
        pd.read_csv(os.path.join(data_dir, f'model_selection_{label}_msl_lt.csv')),
    ])
    grouped = data.groupby(['name', 'estimator', 'params']).agg({score: 'mean'}).reset_index(drop=False)
    best_params = grouped.groupby(['name']).apply(lambda pdf: pdf.sort_values(score).iloc[-1]).reset_index(drop=True)
    best_params['name'] = best_params['name'].apply(lambda x: inverse_map.get(x, x))
    best_params = best_params[['name', 'params', score]]
    best_params.to_csv(os.path.join(data_dir, f'params_{label}_lt.csv'), index=False)

print("=== Comparing '<=' -selected vs '<' -selected hyperparameters and scores ===", flush=True)

for label in labels:
    score = 'auc' if label[-1] == 'c' else 'r2'
    leq = pd.read_csv(os.path.join(data_dir, f'params_{label}.csv'))
    lt = pd.read_csv(os.path.join(data_dir, f'params_{label}_lt.csv'))
    merged = pd.merge(leq, lt, on='name', suffixes=('_leq', '_lt'))
    merged['params_differ'] = merged['params_leq'] != merged['params_lt']
    merged['score_diff'] = merged[f'{score}_leq'] - merged[f'{score}_lt']
    n_differ = merged['params_differ'].sum()
    print(f"\n{label}: {n_differ} / {len(merged)} datasets get DIFFERENT optimal hyperparameters "
          f"when tuned for '<' vs '<='")
    print(merged[['name', 'params_leq', 'params_lt', 'params_differ',
                   f'{score}_leq', f'{score}_lt', 'score_diff']].to_string(index=False))
