import pytest
import numpy as np
from numpy.testing import assert_raises, assert_array_equal

from mcmodels.models import HomogeneousModel


# ============================================================================
# HomogeneousModel class
# ============================================================================
def test_fit():
    # ------------------------------------------------------------------------
    # test columns is set
    X = np.random.rand(10, 10)
    y = np.ones(10)
    reg = HomogeneousModel(kappa=np.inf)

    reg.fit(X, y)

    assert hasattr(reg, 'columns_')


def test_predict():
    # ------------------------------------------------------------------------
    # test correct output
    X = np.ones((10,10))
    X *= np.arange(10)

    columns = np.array([3, 4, 5])

    y = np.zeros((10,10))
    y[:, columns] = X[:, columns]

    reg = HomogeneousModel()
    reg.columns_ = columns
    reg.coef_ = np.eye(10)[:, columns]

    assert_array_equal(y, reg.predict(X))


def test_weights():
    # ------------------------------------------------------------------------
    # test weights are correct shape
    X = np.random.rand(100, 10)
    y = np.random.rand(100, 20)
    reg = HomogeneousModel()
    reg.fit(X, y)

    assert reg.weights.shape == (10, 20)
