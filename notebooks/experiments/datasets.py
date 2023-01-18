import numpy as np
import pandas as pd

import common_datasets.binary_classification as binclas
import common_datasets.regression as regr

__all__ = ['binclas_datasets', 'regr_datasets']

datasets = binclas.get_filtered_data_loaders(n_bounds=(1, 1000), n_minority_bounds=(5, 1000), n_from_phenotypes=1)
names = [dataset()['name'] for dataset in datasets]

summary = binclas.get_summary_pdf()
summary = summary[summary['name'].isin(names)].copy()

summary['n_grid'] = summary[['grid', 'n_feature_uniques']]\
    .apply(lambda x: np.sum(np.array(x['grid']) & (np.array(x['n_feature_uniques']) > 2)), axis=1)

binclas_datasets = summary[summary['n_grid'] > 0].copy()
binclas_datasets = binclas_datasets[~binclas_datasets['name'].isin(['iris0', 'dermatology_6'])]
binclas_datasets['total'] = binclas_datasets['n'] * binclas_datasets['n_col']

binclas_datasets = binclas_datasets.sort_values('total')
binclas_datasets = binclas_datasets.iloc[:20, :]

binclas_datasets = binclas_datasets[['name', 'citation_key', 'n_col', 'n', 'n_minority', 'n_grid', 'data_loader_function']].reset_index(drop=True)

datasets = regr.get_filtered_data_loaders(n_bounds=(1, 2000), n_from_phenotypes=1)
names = [dataset()['name'] for dataset in datasets]

summary = regr.get_summary_pdf()
summary = summary[summary['name'].isin(names)].copy()

summary['n_grid'] = summary[['grid', 'n_feature_uniques']]\
    .apply(lambda x: np.sum(np.array(x['grid']) & (np.array(x['n_feature_uniques']) > 2)), axis=1)

regr_datasets = summary[summary['n_grid'] > 0].copy()
regr_datasets['total'] = regr_datasets['n'] * regr_datasets['n_col']

regr_datasets = regr_datasets.sort_values('total')
regr_datasets = regr_datasets.iloc[:20, :]

regr_datasets = regr_datasets[['name', 'citation_key', 'n_col', 'n', 'n_grid', 'data_loader_function']].reset_index(drop=True)
