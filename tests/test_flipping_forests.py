"""
This module tests the flipping forest
"""
import numpy as np

import pytest

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from flipping_random_forest import (FlippingRandomForestClassifier,
                                    FlippingRandomForestRegressor,
                                    RandomStateMixin,
                                    FlippingBaggingBase,
                                    fit_forest)

dataset = load_diabetes()
X_reg = dataset['data']
y_reg = dataset['target']

dataset = load_breast_cancer()
X_clf = dataset['data']
y_clf = dataset['target']

def test_fit_forest():
    """
    Testing the fit forest function
    """
    estimator = fit_forest({}, X_reg, y_reg, DecisionTreeRegressor)
    pred = estimator.predict(X_reg)

    assert pred.shape[0] == X_reg.shape[0]

def test_random_state_mixin():
    """
    Testing the random state mixin.
    """
    obj = RandomStateMixin(random_state=None)
    assert obj.random_state == np.random

    obj = RandomStateMixin(random_state=5)
    assert isinstance(obj.random_state, np.random.RandomState)
    assert obj.get_params()['random_state'] == 5

    obj = RandomStateMixin(random_state=np.random.RandomState(5))
    assert isinstance(obj.random_state, np.random.RandomState)

    obj = RandomStateMixin(random_state=np.random)
    assert obj.random_state == np.random

    with pytest.raises(ValueError):
        obj = RandomStateMixin(random_state='apple')

    assert len(obj.get_params()) == 1

@pytest.mark.skip("no idea")
def test_flipping_bagging_base():
    """
    Testing the flipping bagging base class
    """
    fbb = FlippingBaggingBase()

    params = fbb.get_params()
    assert len(params) > 0

    fbb.max_features = 'sqrt'
    max_features = fbb.determine_max_features(10)
    assert max_features == 'sqrt'

    fbb.max_features = 'all'
    max_features = fbb.determine_max_features(10)
    assert max_features == 10

    fbb.max_features = 'sqrt-1'
    max_features = fbb.determine_max_features(10)
    assert max_features == 2

    fbb.max_features = 'sqrt-2'
    max_features = fbb.determine_max_features(10)
    assert max_features == 1

    fbb.max_features = 'sqrt+1'
    max_features = fbb.determine_max_features(10)
    assert max_features == 4

    fbb.max_features = 5
    max_features = fbb.determine_max_features(10)
    assert max_features == 5

    fbb.flipping = None
    flip = fbb.flip(np.random.random_sample(size=(10, 3)))
    assert flip == 1

    fbb.flipping = 'coordinate'
    flip = fbb.flip(np.random.random_sample(size=(10, 3)))
    assert np.all((flip == -1) | (flip == 1))

    fbb.flipping = 'full'
    flip = fbb.flip(np.random.random_sample(size=(10, 3)))
    assert flip == -1 or flip == 1

    with pytest.raises(ValueError):
        fbb.bootstrap_sampling(None, None, None)

    with pytest.raises(ValueError):
        fbb.fit_1(None, None)

def test_predict_predict_proba():
    """
    Testing the predict and predict proba functions
    """

    clf = FlippingRandomForestClassifier(random_state=5)\
                .fit(X_clf, y_clf)

    proba = clf.predict_proba(X_clf)
    pred = clf.predict(X_clf)

    np.testing.assert_array_equal(pred, (proba[:, 1] > 0.5).astype(int))

def test_flipping_forest_classifier():
    """
    Testing the flipping forest classifier
    """

    auc_orig = []
    auc_flip = []
    auc_flip_n = []
    auc_flip_sw = []

    validator = RepeatedStratifiedKFold(n_splits=5,
                                        n_repeats=5,
                                        random_state=5)

    for train, test in validator.split(X_clf, y_clf, y_clf):
        X_train = X_clf[train]
        X_test = X_clf[test]
        y_train = y_clf[train]
        y_test = y_clf[test]

        pred = RandomForestClassifier(random_state=5)\
                .fit(X_train, y_train)\
                .predict_proba(X_test)[:, 1]

        auc_orig.append(roc_auc_score(y_test, pred))

        pred = FlippingRandomForestClassifier(random_state=5)\
                .fit(X_train, y_train)\
                .predict_proba(X_test)[:, 1]

        auc_flip.append(roc_auc_score(y_test, pred))

        pred = FlippingRandomForestClassifier(random_state=5, n_jobs=2)\
                .fit(X_train, y_train)\
                .predict_proba(X_test)[:, 1]

        auc_flip_n.append(roc_auc_score(y_test, pred))

        pred = FlippingRandomForestClassifier(random_state=5)\
                .fit(X_train, y_train, sample_weight=np.repeat(1.0, X_train.shape[0]))\
                .predict_proba(X_test)[:, 1]

        auc_flip_sw.append(roc_auc_score(y_test, pred))

    auc_orig = np.mean(auc_orig)
    auc_flip = np.mean(auc_flip)
    auc_flip_n = np.mean(auc_flip_n)
    auc_flip_sw = np.mean(auc_flip_sw)

    diff = np.abs(auc_orig - auc_flip)
    diff_n = np.abs(auc_orig - auc_flip_n)
    diff_sw = np.abs(auc_orig - auc_flip_sw)

    assert diff / auc_orig < 0.1
    assert diff_n / auc_orig < 0.1
    assert diff_sw / auc_orig < 0.1

def test_flipping_forest_regressor():
    """
    Testing the flipping forest regressor
    """

    r2_orig = []
    r2_flip = []
    r2_flip_n = []
    r2_flip_sw = []

    validator = RepeatedKFold(n_splits=5,
                                n_repeats=5,
                                random_state=5)

    for train, test in validator.split(X_reg, y_reg):
        X_train = X_clf[train]
        X_test = X_clf[test]
        y_train = y_clf[train]
        y_test = y_clf[test]

        pred = RandomForestRegressor(random_state=5)\
                .fit(X_train, y_train)\
                .predict(X_test)

        r2_orig.append(r2_score(y_test, pred))

        pred = FlippingRandomForestRegressor(random_state=5)\
                .fit(X_train, y_train)\
                .predict(X_test)

        r2_flip.append(r2_score(y_test, pred))

        pred = FlippingRandomForestRegressor(random_state=5, n_jobs=2)\
                .fit(X_train, y_train)\
                .predict(X_test)

        r2_flip_n.append(r2_score(y_test, pred))

        pred = FlippingRandomForestRegressor(random_state=5)\
                .fit(X_train, y_train, sample_weight=np.repeat(1.0, X_train.shape[0]))\
                .predict(X_test)

        r2_flip_sw.append(r2_score(y_test, pred))

    r2_orig = np.mean(r2_orig)
    r2_flip = np.mean(r2_flip)
    r2_flip_n = np.mean(r2_flip_n)
    r2_flip_sw = np.mean(r2_flip_sw)

    diff = np.abs(r2_orig - r2_flip)
    diff_n = np.abs(r2_orig - r2_flip_n)
    diff_sw = np.abs(r2_orig - r2_flip_sw)

    assert diff / r2_orig < 0.1
    assert diff_n / r2_orig < 0.1
    assert diff_sw / r2_orig < 0.1
