import pytest
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_array_almost_equal

from mcmodels.regressors import nonnegative_ridge_regression, NonnegativeRidge

# ============================================================================
# Module level functions
# ============================================================================
def test_nonnegative_ridge_regression():
    # ------------------------------------------------------------------------
    # test shape incompatibility
    X, y = np.ones((10, 1)), np.ones((11, 1))
    alpha = np.zeros(1)

    assert_raises(ValueError, nonnegative_ridge_regression, X, y, alpha)

    # ------------------------------------------------------------------------
    # test X.ndim != 2
    X = np.linspace(-10, 10, 100)
    y = 4*X

    assert_raises(ValueError, nonnegative_ridge_regression, X, y, alpha)

    # ------------------------------------------------------------------------
    # test incompatible alpha shape
    X = X.reshape(-1, 1)
    alpha = np.zeros(2)

    assert_raises(ValueError, nonnegative_ridge_regression, X, y, alpha)

    # ------------------------------------------------------------------------
    # test sample_weight shape incompatibility
    sample_weight = np.ones(11)

    assert_raises(ValueError, nonnegative_ridge_regression, X, y, alpha, sample_weight)


# ============================================================================
# NonnegativeRidge class
# ============================================================================
def test_fit():
    # ------------------------------------------------------------------------
    # test sample_weight shape incompatibility
    reg = NonnegativeRidge()
    X = np.linspace(-10, 10, 100)
    y = 4*X
    sample_weight = np.ones(11)

    assert_raises(ValueError, reg.fit, X, y, sample_weight)

    sample_weight = np.ones((10, 1))
    assert_raises(ValueError, reg.fit, X, y, sample_weight)
