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
    params,
    validator_params,
    random_state):

    results = []

    random_state = np.random.RandomState(random_state)

    for ddx, row in datasets.iterrows():
        dataset = row['data_loader_function']()

        print(datetime.datetime.now(), dataset['name'])

        X = dataset['data']
        y = dataset['target']

        mask = np.arange(X.shape[0])
        random_state.shuffle(mask)
        X = X[mask]
        y = y[mask]

        validator = RepeatedStratifiedKFold(**validator_params)

        for fdx, (train, test) in enumerate(validator.split(X, y, y)):
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]

            if isinstance(params, list):
                params_tmp = params
            elif isinstance(params, dict):
                params_tmp = [params[dataset['name']]]

            for idx, param in enumerate(params_tmp):
                classifier = estimator(**param)
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict_proba(X_test)[:, 1]

                if len(y_pred.shape) == 2:
                    y_pred = y_pred[:, 1]

                results.append({'name': dataset['name'],
                                'fold': fdx,
                                'auc': roc_auc_score(y_test, y_pred),
                                'params': param,
                                'estimator': estimator.__name__})

    return pd.DataFrame(results)

def evaluate_regression(*,
    datasets: pd.DataFrame,
    estimator,
    params,
    validator_params,
    random_state):

    results = []

    random_state = np.random.RandomState(random_state)

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

            if isinstance(params, list):
                params_tmp = params
            elif isinstance(params, dict):
                params_tmp = [params[dataset['name']]]

            for idx, param in enumerate(params_tmp):
                classifier = estimator(**param)
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)

                results.append({'name': dataset['name'],
                                'fold': fdx,
                                'r2': r2_score(y_test, y_pred),
                                'params': param,
                                'estimator': estimator.__name__})

    return pd.DataFrame(results)