"""
This module tests the flipping forest
"""
import numpy as np

import pytest

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

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

def test_flipping_bagging_base():
    """
    Testing the flipping bagging base class
    """
