import pytest
import numpy as np
from numpy.testing import assert_raises, assert_allclose, assert_array_almost_equal

from mcmodels.regressors import NonnegativeElasticNet, nonnegative_elastic_net_regression
from mcmodels.regressors.nonnegative_linear.elastic_net import _solve_elastic_net_nnls

# ============================================================================
# Module level functions
# ============================================================================
def test_solve_elastic_net_nnls():
    # ------------------------------------------------------------------------
    # test output (with alpha=0, should equal core._solve_nnls)
    X = np.linspace(-10, 10, 100).reshape(-1, 1)
    y = 4*X

    coef, res = _solve_elastic_net_nnls(X, y)

    assert_allclose(coef[0], 4.)
    assert_allclose(res[0], 0.0, atol=1e-10)


def test_nonnegative_elastic_net_regression():
    # ------------------------------------------------------------------------
    # test shape incompatibility
    X, y = np.ones((10, 1)), np.ones((11, 1))
    alpha = np.zeros(1)
    rho = np.zeros(1)

    assert_raises(ValueError, nonnegative_elastic_net_regression, X, y, alpha, rho)

    # ------------------------------------------------------------------------
    # test X.ndim != 2
    X = np.linspace(-10, 10, 100)
    y = 4*X

    assert_raises(ValueError, nonnegative_elastic_net_regression, X, y, alpha, rho)

    # ------------------------------------------------------------------------
    # test incompatible alpha shape
    X = X.reshape(-1, 1)
    alpha = np.zeros(2)
    rho = np.zeros(2)

    assert_raises(ValueError, nonnegative_elastic_net_regression, X, y, alpha, rho)

    # ------------------------------------------------------------------------
    # test sample_weight
    alpha = np.arange(X.shape[1])
    rho = np.arange(X.shape[1])
    sample_weight = 1.0

    coef, res = _solve_elastic_net_nnls(X, y.reshape(-1, 1), alpha, rho)
    coef_sw, res_sw = nonnegative_elastic_net_regression(X, y, alpha, rho, sample_weight)

    assert_array_almost_equal(coef.ravel(), coef_sw)
    assert_array_almost_equal(res, res_sw)

    # ------------------------------------------------------------------------
    # test sample_weight shape incompatibility
    sample_weight = np.ones(11)

    assert_raises(ValueError, nonnegative_elastic_net_regression, X, y, alpha, rho, sample_weight)


# ============================================================================
# NonnegativeElasticNet class
# ============================================================================
def test_fit():
    # ------------------------------------------------------------------------
    # test sample_weight shape incompatibility
    reg = NonnegativeElasticNet()
    X = np.linspace(-10, 10, 100)
    y = 4*X
    sample_weight = np.ones(11)

    assert_raises(ValueError, reg.fit, X, y, sample_weight)

    sample_weight = np.ones((10, 1))
    assert_raises(ValueError, reg.fit, X, y, sample_weight)
