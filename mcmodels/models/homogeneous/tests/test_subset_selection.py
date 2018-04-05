import pytest
import numpy as np
from numpy.testing import assert_raises, assert_array_equal

from mcmodels.models.homogeneous \
    import svd_subset_selection, condition_with_svd_subset_selection

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


def test_condition_X():
    # ------------------------------------------------------------------------
    # test X is not overwritten
    X = np.random.rand(10, 10)
    kappa = np.inf

    X_condioned, columns = condition_with_svd_subset_selection(X, kappa)

    assert X_condioned is not X
    assert_array_equal(X, X_condioned)
