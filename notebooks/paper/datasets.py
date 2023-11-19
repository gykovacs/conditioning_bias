"""
Imports and filters the datasets used in the experiments
"""

import logging

import numpy as np
import pandas as pd

import common_datasets.binary_classification as binclas
import common_datasets.regression as regr

from config import (n_datasets,
                    n_bounds_binclas,
                    n_bounds_regr,
                    n_from_phenotypes,
                    n_minority_bounds_binclas)

__all__ = ['binclas_datasets', 'regr_datasets']

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# determining the binary classification datasets containing grid features

logging.info("querying the filtered classification datasets")

datasets = binclas.get_filtered_data_loaders(n_bounds=n_bounds_binclas,
                                                n_minority_bounds=n_minority_bounds_binclas,
                                                n_from_phenotypes=n_from_phenotypes)

logging.info("ranking the datasets")

summary = binclas.get_summary_pdf()
summary = summary[summary['data_loader_function'].isin(datasets)].copy()

summary['n_grid'] = summary[['grid', 'n_feature_uniques']]\
    .apply(lambda x: np.sum(np.array(x['grid'])), axis=1)

binclas_datasets = summary[summary['n_grid'] > 0].copy()

# dropping the datasets which are separable
binclas_datasets = binclas_datasets[~binclas_datasets['name'].isin(['iris0', 'dermatology_6'])]
binclas_datasets['total'] = binclas_datasets['n'] * binclas_datasets['n_col']

binclas_datasets = binclas_datasets.sort_values('total')
binclas_datasets = binclas_datasets.iloc[:n_datasets, :]

binclas_datasets = binclas_datasets[['name',
                                        'citation_key',
                                        'n_col',
                                        'n',
                                        'n_minority',
                                        'n_grid',
                                        'data_loader_function']].reset_index(drop=True)

logging.info("binary classification datasets prepared")

# determining the regression datasets containing grid features

logging.info("querying the filtered regression datasets")

datasets = regr.get_filtered_data_loaders(n_bounds=n_bounds_regr,
                                            n_from_phenotypes=n_from_phenotypes)

logging.info("ranking the datasets")

summary = regr.get_summary_pdf()
summary = summary[summary['data_loader_function'].isin(datasets)].copy()

summary['n_grid'] = summary[['grid', 'n_feature_uniques']]\
    .apply(lambda x: np.sum(np.array(x['grid'])), axis=1)

regr_datasets = summary[summary['n_grid'] > 0].copy()
regr_datasets['total'] = regr_datasets['n'] * regr_datasets['n_col']

regr_datasets = regr_datasets.sort_values('total')
regr_datasets = regr_datasets.iloc[:n_datasets, :]

regr_datasets = regr_datasets[['name',
                                'citation_key',
                                'n_col',
                                'n',
                                'n_grid',
                                'grid',
                                'data_loader_function']].reset_index(drop=True)

logging.info("regression datasets prepared")
