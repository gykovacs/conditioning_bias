"""
Testing the ordinary regressors
"""

import pytest

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from flipping_random_forest import (
    OperatorDecisionTreeRegressor,
    OperatorRandomForestRegressor,
    determine_specific_operator_regression
)

@pytest.mark.parametrize('random_state', [2, 3, 4, 5])
def test_identity_dt(random_state):
    """
    Testing if the regressors lead to identical results with <=
    """

    random_state = np.random.RandomState(random_state)

    X = random_state.randint(0, 10, size=(100, 3))
    X_test = random_state.randint(0, 10, size=(20, 3))
    y = random_state.random_sample(size=100)

    dt = DecisionTreeRegressor(random_state=5).fit(X, y)
    odt = OperatorDecisionTreeRegressor(mode='<=', random_state=5).fit(X, y)

    pred_sk = dt.predict(X_test)
    pred = odt.predict(X_test)

    np.testing.assert_array_equal(pred_sk, pred)

@pytest.mark.parametrize('random_state', [2, 3, 4, 5])
def test_identity_rf(random_state):
    """
    Testing if the regressors lead to identical results with <=
    """

    random_state = np.random.RandomState(random_state)

    X = random_state.randint(0, 10, size=(100, 3))
    X_test = random_state.randint(0, 10, size=(20, 3))
    y = random_state.random_sample(size=100)

    rf = RandomForestRegressor(random_state=5).fit(X, y)
    orf = OperatorRandomForestRegressor(mode='<=', random_state=5).fit(X, y)

    pred_sk = rf.predict(X_test)
    pred = orf.predict(X_test)

    np.testing.assert_array_equal(pred_sk, pred)

@pytest.mark.parametrize('random_state', [2, 3, 4, 5])
def test_modes_dt(random_state):
    """
    Testing if the dt classifiers work with various modes
    """

    random_state = np.random.RandomState(random_state)

    X = random_state.randint(0, 10, size=(100, 3))
    X_test = random_state.randint(0, 10, size=(20, 3))
    y = random_state.random_sample(size=100)

    odt = OperatorDecisionTreeRegressor(mode='<=', random_state=5).fit(X, y)

    for mode in odt.get_modes():
        odt.set_mode(mode)
        assert odt.predict(X_test).shape == (20,)

@pytest.mark.parametrize('random_state', [2, 3, 4, 5])
def test_modes_rf(random_state):
    """
    Testing if the rf classifiers work with various modes
    """

    random_state = np.random.RandomState(random_state)

    X = random_state.randint(0, 10, size=(100, 3))
    X_test = random_state.randint(0, 10, size=(20, 3))
    y = random_state.random_sample(size=100)

    odt = OperatorRandomForestRegressor(mode='<=', random_state=5).fit(X, y)

    for mode in odt.get_modes():
        odt.set_mode(mode)
        assert odt.predict(X_test).shape == (20,)

@pytest.mark.parametrize('random_state', [2, 3, 4, 5])
def test_specific(random_state):
    """
    Testing if the specific operator is determined correctly
    """

    random_state = np.random.RandomState(random_state)

    X = random_state.random_sample(size=(100, 3))
    y = random_state.random_sample(size=100)

    assert determine_specific_operator_regression(X, y, None) == '<='
