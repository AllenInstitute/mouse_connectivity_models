import pytest
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_array_almost_equal

from voxel_model.regressors.nonnegative_linear.base import _solve_nnls
from voxel_model.regressors import nonnegative_regression, NonnegativeLinear

# ============================================================================
# Module level functions
# ============================================================================

def test_solve_nnls():
    # ------------------------------------------------------------------------
    # test X.ndim != 2 or y.ndim != 2
    X = np.linspace(-10, 10, 100)
    y = 4*X

    assert_raises(ValueError, _solve_nnls, X, y)

    # ------------------------------------------------------------------------
    # test incompatible alpha shape
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)

    coef, res = _solve_nnls(X, y)

    assert_allclose(coef[0], 4.)
    assert_allclose(res[0], 0.0, atol=1e-10)


def test_nonnegative_regression():
    # ------------------------------------------------------------------------
    # test shape incompatibility
    X, y = np.ones((10, 1)), np.ones((11, 1))

    assert_raises(ValueError, nonnegative_regression, X, y)

    # ------------------------------------------------------------------------
    # test X.ndim != 2
    X = np.linspace(-10, 10, 100)
    y = 4*X

    assert_raises(ValueError, nonnegative_regression, X, y)

    # ------------------------------------------------------------------------
    # test function output
    X = X.reshape(-1, 1)

    coef, res = nonnegative_regression(X, y)

    assert_allclose(coef[0], 4.)
    assert_allclose(res[0], 0.0, atol=1e-10)

    # ------------------------------------------------------------------------
    # test sample_weight
    sample_weight = 1.0

    coef, res = nonnegative_regression(X, y)
    coef_sw, res_sw = nonnegative_regression(X, y, sample_weight)

    assert_array_almost_equal(coef.ravel(), coef_sw)
    assert_array_almost_equal(res, res_sw)

    # ------------------------------------------------------------------------
    # test sample_weight shape incompatibility
    sample_weight = np.ones(11)

    assert_raises(ValueError, nonnegative_regression, X, y, sample_weight)


# ============================================================================
# NonnegativeLinear class
# ----------------------------------------------------------------------------

def test_fit():
    # ------------------------------------------------------------------------
    # test sample_weight shape incompatibility
    reg = NonnegativeLinear()
    X = np.linspace(-10, 10, 100)
    y = 4*X
    sample_weight = np.ones(11)

    assert_raises(ValueError, reg.fit, X, y, sample_weight)

    sample_weight = np.ones((10, 1))
    assert_raises(ValueError, reg.fit, X, y, sample_weight)


def test_predict():
    # ------------------------------------------------------------------------
    # test train predict close
    reg = NonnegativeLinear()
    X = np.linspace(-10, 10, 100).reshape(-1, 1)
    y = 4*X

    reg.fit(X, y)

    assert_array_almost_equal(y, reg.predict(X))
