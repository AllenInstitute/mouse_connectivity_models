from __future__ import division
import pytest
from scipy.sparse import csr_matrix
import numpy as np
from numpy.testing \
    import assert_array_equal, assert_array_almost_equal, assert_allclose


from voxel_model.regressors import NadarayaWatson, NadarayaWatsonCV
from voxel_model.regressors.nonparametric.nadaraya_watson \
    import _NadarayaWatsonLOOCV


# ============================================================================
# Nadaraya-Watson class
# ============================================================================
def test_get_kernel():
    # ------------------------------------------------------------------------
    # test that callable can be passed
    from voxel_model.regressors.nonparametric.kernels import Uniform
    kernel = Uniform(support=1e5)
    reg = NadarayaWatson(kernel=kernel)
    X = np.random.rand(10, 3)

    assert_array_equal(np.ones((10,10)), reg._get_kernel(X))


def test_check_fit_arrays():
    # ------------------------------------------------------------------------
    # test sample weight
    reg = NadarayaWatson()
    X = np.random.rand(10, 10)
    y = np.random.rand(10, 3)
    sample_weight = np.random.rand(10)

    X_check, y_check = reg._check_fit_arrays(X, y, sample_weight)

    assert_array_equal(X, X_check)
    assert_array_almost_equal(sample_weight[:, np.newaxis] * y, y_check)


def test_normalize_kernel():
    # ------------------------------------------------------------------------
    # test overwrite/not
    reg = NadarayaWatson()
    K = np.ones((5, 5))
    K_norm = K / 5

    assert_array_equal(K_norm, reg._normalize_kernel(K, overwrite=False))
    assert K_norm is reg._normalize_kernel(K_norm, overwrite=True)

    # ------------------------------------------------------------------------
    # test handling of zero rows
    K_norm[2, :] = np.zeros(5)

    assert_array_equal(K_norm, reg._normalize_kernel(K_norm, overwrite=False))


def test_nw_fit():
    # ------------------------------------------------------------------------
    # test setting of attrs
    reg = NadarayaWatson()
    X = np.random.rand(10, 10)
    y = np.random.rand(10, 3)

    reg.fit(X, y)

    assert X is reg.X_
    assert y is reg.y_


def test_get_weights():
    # ------------------------------------------------------------------------
    # test weights are row normalized
    reg = NadarayaWatson()
    X = np.random.rand(10, 3)
    y = np.random.rand(10, 10)

    w = reg.fit(X, y).get_weights(X)

    assert_allclose(w.sum(axis=1), 1.)


def test_predict():
    # ------------------------------------------------------------------------
    # test sparse predict == dense predict
    reg = NadarayaWatson()
    X = np.random.rand(10, 3)
    y = np.random.rand(10, 10)

    dense = reg.fit(X, y).predict(X)
    sparse_y = reg.fit(X, csr_matrix(y)).predict(X)
    sparse_x = reg.fit(csr_matrix(X), y).predict(X)
    sparse_p = reg.fit(X, y).predict(csr_matrix(X))

    assert_array_almost_equal(dense, sparse_y)
    assert_array_almost_equal(dense, sparse_x)
    assert_array_almost_equal(dense, sparse_p)


# ============================================================================
# _NadarayaWatsonLOOCV class
# ============================================================================
def test_errors_and_values_helper():
    # ------------------------------------------------------------------------
    # test diag(K) == 0, normalized
    param_grid = dict(kernel=['rbf', 'linear'])
    reg = _NadarayaWatsonLOOCV(param_grid)
    K = np.ones((5, 5))

    S = reg._errors_and_values_helper(K)

    assert_allclose(np.diagonal(S), 0.)
    assert_allclose(S.sum(axis=1), 1.)


def test_errors():
    # ------------------------------------------------------------------------
    # test returns loocv errors
    param_grid = dict(kernel=['rbf', 'linear'])
    reg = _NadarayaWatsonLOOCV(param_grid)
    K = np.ones((5, 5))
    y = np.ones(5).reshape(-1,1)

    assert reg._errors(K, y) == 0.


def test_values():
    # ------------------------------------------------------------------------
    # test returns loocv prediction
    param_grid = dict(kernel=['rbf', 'linear'])
    reg = _NadarayaWatsonLOOCV(param_grid)
    K = np.ones((5, 5))
    y = np.ones(5).reshape(-1,1)

    assert_array_equal(y, reg._values(K, y))


def test_nwloocv_fit():
    # ------------------------------------------------------------------------
    # test attrs set
    param_grid = dict(kernel=['rbf', 'linear'])
    reg = _NadarayaWatsonLOOCV(param_grid, store_cv_scores=True)
    X = np.random.rand(10, 3)
    y = np.random.rand(10, 10)

    reg.fit(X, y)

    assert hasattr(reg, 'n_splits_')
    assert hasattr(reg, 'best_index_')
    assert hasattr(reg, 'best_score_')
    assert hasattr(reg, 'best_params_')
    assert hasattr(reg, 'cv_scores_')


# ============================================================================
# NadarayaWatsonCV class
# ============================================================================
def test_update_params():
    # ------------------------------------------------------------------------
    # test update params
    param_grid = dict(kernel=['rbf', 'linear'])
    reg = NadarayaWatsonCV(param_grid)

    params = dict(a='a')
    reg._update_params(params)

    assert hasattr(reg, 'a')


def test_nwcv_fit():
    # ------------------------------------------------------------------------
    # test attrs set
    param_grid = dict(kernel=['rbf', 'linear'])
    reg = NadarayaWatsonCV(param_grid, store_cv_scores=True)
    X = np.random.rand(10, 3)
    y = np.random.rand(10, 10)

    reg.fit(X, y)

    assert hasattr(reg, 'n_splits_')
    assert hasattr(reg, 'best_index_')
    assert hasattr(reg, 'best_score_')
    assert hasattr(reg, 'cv_scores_')

    # ------------------------------------------------------------------------
    # test we get same result with GridsearchCV & LeaveOneOut
    from sklearn.model_selection import LeaveOneOut
    reg_grid = NadarayaWatsonCV(param_grid, cv=LeaveOneOut(),
                                scoring='neg_mean_squared_error')

    reg_grid.fit(X, y)

    assert reg.best_score_ == pytest.approx(reg_grid.best_score_)
    assert reg.kernel == reg_grid.kernel
    assert reg.n_splits_ == reg_grid.n_splits_
