"""
NOTE :: NEED to add testing for __init__ esp for passing fitted voxel_model
"""
from __future__ import division
import pytest
import numpy as np
from numpy.testing \
    import assert_array_equal, assert_array_almost_equal, assert_raises

from voxel_model.implicit_construction import ImplicitModel

@pytest.fixture(scope="module")
def weights():
    return np.array([[5, 3, 2],
                     [7, 2, 1],
                     [1, 4, 5],
                     [2, 6, 2],
                     [3, 4, 3]])

@pytest.fixture(scope="module")
def nodes():
    return np.array([[1, 2, 1, 4, 5, 6, 2, 1],
                     [0, 0, 1, 2, 0, 0, 0, 1],
                     [2, 2, 2, 1, 0, 7, 6, 9]])

@pytest.fixture(scope="module")
def true_model(weights, nodes):
    return weights.dot(nodes)

@pytest.fixture(scope="module")
def implicit_model(weights, nodes):
    return ImplicitModel(weights=weights, nodes=nodes)

# ============================================================================
# constructors
# ----------------------------------------------------------------------------
# test
def test_from_hdf5():
    args = ("weights.hdf5", "nodes.hdf5")
    assert_raises( NotImplementedError, ImplicitModel.from_hdf5, *args )

# ----------------------------------------------------------------------------
# test
def test_from_csv():
    pass

# ----------------------------------------------------------------------------
# test
def test_from_npy():
    pass

# ----------------------------------------------------------------------------
# test
def test_from_fitted_voxel_model():
    pass

# ============================================================================
# dunder methods
# ----------------------------------------------------------------------------
# test
def test_init():
    pass

# ----------------------------------------------------------------------------
# test
def test_getitem():
    pass

# ----------------------------------------------------------------------------
# test
def test_len():
    pass

# ============================================================================
# properties
# ----------------------------------------------------------------------------
# test
def test_dtype():
    pass

# ----------------------------------------------------------------------------
# test
def test_shape():
    pass

# ----------------------------------------------------------------------------
# test
def test_size():
    pass

# ----------------------------------------------------------------------------
# test
def test_T():
    pass

# ============================================================================
# methods
# ----------------------------------------------------------------------------
# test
def test_transpose():
    pass

# ----------------------------------------------------------------------------
# test
def test_astype():
    pass

# ----------------------------------------------------------------------------
# test
def test_sum():
    pass

# ----------------------------------------------------------------------------
# test
def test_mean():
    pass

# ----------------------------------------------------------------------------
# test
def test_iterrows(true_model, implicit_model):
    for i, row in enumerate(implicit_model.iterrows()):
        assert_array_equal( row, true_model[i] )

# ----------------------------------------------------------------------------
# test
def test_itercolumns(true_model, implicit_model):
    for j, column in enumerate(implicit_model.itercolumns()):
        assert_array_equal( column, true_model[:, j] )

# ----------------------------------------------------------------------------
# test
def test_iterrow_blocks():
    pass

# ----------------------------------------------------------------------------
# test
def test_itercolumns_blocks():
    pass
