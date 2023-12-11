"""
This module implements a general purpose evaluation method
"""

import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold
)

from sklearn.metrics import roc_auc_score, r2_score

from config import (n_repeats, n_splits)

def evaluate_classification(*,
    datasets: pd.DataFrame,
    estimator,
    modes: list,
    params,
    validator_params: dict,
    random_seed: int):
    """
    The heavy-lifting cross-validation function for classification

    Args:
        datasets (pd.DataFrame): the dataset summary
        estimator (class): the estimator class to be used
        modes (list): the mode parameters to be used for evaluation
        params (list|dict): the list of parameterizations or a dictionary of
                            dataset specific parametrizations
        validator_params (dict): the validator parameters to be used
        random_seed (int): the random seed to be used

    Returns:
        pd.DataFrame: the results of the evaluation
    """

    results = []

    random_state = np.random.RandomState(random_seed)

    for ddx, row in datasets.iterrows():
        dataset = row['data_loader_function']()

        print(datetime.datetime.now(), dataset['name'])

        X = dataset['data']
        y = dataset['target']

        mask = np.arange(X.shape[0])
        random_state.shuffle(mask)
        X = X[mask]
        y = y[mask]

        vp_tmp = validator_params.copy()
        if vp_tmp['n_splits'] > np.sum(y):
            vp_tmp['n_splits'] = np.sum(y)

        validator = RepeatedStratifiedKFold(**vp_tmp)

        for fdx, (train, test) in enumerate(validator.split(X, y, y)):
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]

            if isinstance(params, dict):
                params_list = [params[dataset['name']] | {'random_state': random_seed}]
            else:
                params_list = params

            for par in params_list:
                classifier = estimator(**par)
                classifier.fit(X_train, y_train)

                for idx, mode in enumerate(modes):
                    classifier.set_mode(mode)
                    y_pred = classifier.predict_proba(X_test)[:, 1]

                    results.append({'name': dataset['name'],
                                    'fold': fdx,
                                    'auc': roc_auc_score(y_test, y_pred),
                                    'mode': mode,
                                    'params': par,
                                    'estimator': estimator.__name__})

    return pd.DataFrame(results)

def evaluate_regression(*,
    datasets: pd.DataFrame,
    estimator,
    modes,
    params,
    validator_params,
    random_seed):

    results = []

    random_state = np.random.RandomState(random_seed)

    for ddx, row in datasets.iterrows():
        dataset = row['data_loader_function']()

        print(datetime.datetime.now(), dataset['name'])

        X = dataset['data']
        y = dataset['target']
        validator = RepeatedKFold(**validator_params)

        for fdx, (train, test) in enumerate(validator.split(X)):
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]

            if isinstance(params, dict):
                params_list = [params[dataset['name']] | {'random_state': random_seed}]
            else:
                params_list = params

            for par in params_list:
                regressor = estimator(**par)
                regressor.fit(X_train, y_train)

                for idx, mode in enumerate(modes):
                    regressor.set_mode(mode)
                    y_pred = regressor.predict(X_test)

                    results.append({'name': dataset['name'],
                                    'fold': fdx,
                                    'r2': r2_score(y_test, y_pred),
                                    'mode': mode,
                                    'params': par,
                                    'estimator': estimator.__name__})

    return pd.DataFrame(results)
