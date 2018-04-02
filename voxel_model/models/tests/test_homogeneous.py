import pytest
import numpy as np
from numpy.testing import assert_raises, assert_array_equal

from voxel_model.models.homogeneous import svd_subset_selection, HomogeneousModel

# ============================================================================
# Module level functions
# ============================================================================
def test_svd_subset_selection():
    # ------------------------------------------------------------------------
    # test ValueError from n < 1 or n > cols
    X = np.random.rand(10,9)

    assert_raises(ValueError, svd_subset_selection, X, 0)
    assert_raises(ValueError, svd_subset_selection, X, 10)

    # ------------------------------------------------------------------------
    # test method removes correct columns
    linear_combination = 2*X[:, 2] - 3*X[:, 8]
    X_singular = np.hstack((X, linear_combination[:, np.newaxis]))

    columns = svd_subset_selection(X_singular, 9)
    removed_column = set(range(10)) - set(columns)

    assert removed_column.pop() in (2, 8, 9)


# ============================================================================
# HomogeneousModel class
# ============================================================================
def test_condition_X():
    # ------------------------------------------------------------------------
    # test X is not overwritten
    X = np.random.rand(10, 10)
    reg = HomogeneousModel(kappa=np.inf)

    X_condioned, columns = reg._condition_X(X)

    assert X_condioned is not X
    assert_array_equal(X, X_condioned)


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
