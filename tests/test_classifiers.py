"""
Testing the ordinary classifiers
"""

import pytest

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from conditioning_bias import (
    OperatorDecisionTreeClassifier,
    OperatorRandomForestClassifier
)

@pytest.mark.parametrize('random_state', [2, 3, 4, 5])
def test_identity_dt(random_state):
    """
    Testing if the dt classifiers lead to identical results with <=
    """

    random_state = np.random.RandomState(random_state)

    X = random_state.randint(0, 10, size=(100, 3))
    X_test = random_state.randint(0, 10, size=(20, 3))
    y = random_state.randint(0, 2, size=100)

    dt = DecisionTreeClassifier(random_state=5).fit(X, y)
    odt = OperatorDecisionTreeClassifier(mode='<=', random_state=5).fit(X, y)

    pred_sk = dt.predict_proba(X_test)
    pred = odt.predict_proba(X_test)

    np.testing.assert_array_equal(pred_sk, pred)

    pred_sk = dt.predict(X_test)
    pred = odt.predict(X_test)

    np.testing.assert_array_equal(pred_sk, pred)

@pytest.mark.parametrize('random_state', [2, 3, 4, 5])
def test_identity_rf(random_state):
    """
    Testing if the rf classifiers lead to identical results with <=
    """

    random_state = np.random.RandomState(random_state)

    X = random_state.randint(0, 10, size=(100, 3))
    X_test = random_state.randint(0, 10, size=(20, 3))
    y = random_state.randint(0, 2, size=100)

    rf = RandomForestClassifier(random_state=5).fit(X, y)
    orf = OperatorRandomForestClassifier(mode='<=', random_state=5).fit(X, y)

    pred_sk = rf.predict_proba(X_test)
    pred = orf.predict_proba(X_test)

    np.testing.assert_array_equal(pred_sk, pred)

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
    y = random_state.randint(0, 2, size=100)

    odt = OperatorDecisionTreeClassifier(mode='<=', random_state=5).fit(X, y)

    for mode in odt.get_modes():
        odt.set_mode(mode)
        assert odt.predict_proba(X_test).shape == (20, 2)

@pytest.mark.parametrize('random_state', [2, 3, 4, 5])
def test_modes_rf(random_state):
    """
    Testing if the rf classifiers work with various modes
    """

    random_state = np.random.RandomState(random_state)

    X = random_state.randint(0, 10, size=(100, 3))
    X_test = random_state.randint(0, 10, size=(20, 3))
    y = random_state.randint(0, 2, size=100)

    odt = OperatorRandomForestClassifier(mode='<=', random_state=5).fit(X, y)

    for mode in odt.get_modes():
        odt.set_mode(mode)
        assert odt.predict_proba(X_test).shape == (20, 2)
