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
    modes,
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

            classifier = estimator(random_state=5)
            classifier.fit(X_train, y_train)

            for idx, mode in enumerate(modes):
                classifier.set_mode(mode)
                y_pred = classifier.predict_proba(X_test)[:, 1]

                results.append({'name': dataset['name'],
                                'fold': fdx,
                                'auc': roc_auc_score(y_test, y_pred),
                                'mode': mode,
                                'estimator': estimator.__name__})

    return pd.DataFrame(results)

def evaluate_regression(*,
    datasets: pd.DataFrame,
    estimator,
    modes,
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

            regressor = estimator(random_state=5)
            regressor.fit(X_train, y_train)

            for idx, mode in enumerate(modes):
                regressor.set_mode(mode)
                y_pred = regressor.predict(X_test)

                results.append({'name': dataset['name'],
                                'fold': fdx,
                                'r2': r2_score(y_test, y_pred),
                                'mode': mode,
                                'estimator': estimator.__name__})

    return pd.DataFrame(results)
