"""
NOTE :: NEED to add testing for __init__ esp for passing fitted true_array
"""
from __future__ import division
import pytest
import numpy as np
from numpy.testing \
    import assert_array_equal, assert_array_almost_equal, assert_raises

from voxel_model.array import Array

@pytest.fixture(scope="module")
def weights():
    return np.random.choice(4, size=(10,5) )

@pytest.fixture(scope="module")
def nodes():
    # more probability of zero (more realistic)
    return np.random.choice(4, size=(5,20), p=(.7,.1,.1,.1))

@pytest.fixture(scope="function")
def true_array(weights, nodes):
    return weights.dot(nodes)

@pytest.fixture(scope="function")
def voxel_model(weights, nodes):
    return Array(weights=weights, nodes=nodes)

# ============================================================================
# constructors
# ----------------------------------------------------------------------------
# test
def test_from_hdf5():
    args = ("weights.hdf5", "nodes.hdf5")
    assert_raises( NotImplementedError, Array.from_hdf5, *args )

# ----------------------------------------------------------------------------
# test
def test_from_csv(tmpdir, weights, nodes):
    f1 = tmpdir.join("weights.csv")
    f2 = tmpdir.join("nodes.csv")

    # numpy doesnt like unicode?
    f1, f2 = map(str, (f1, f2))

    np.savetxt(f1, weights, delimiter=",")
    np.savetxt(f2, nodes, delimiter=",")

    voxel_model = Array.from_csv(f1, f2)

    assert_array_equal( weights, voxel_model.weights )
    assert_array_equal( nodes, voxel_model.nodes )

# ----------------------------------------------------------------------------
# test
def test_from_npy(tmpdir, weights, nodes):
    f1 = tmpdir.join("weights.npy")
    f2 = tmpdir.join("nodes.npy")

    # numpy doesnt like unicode?
    f1, f2 = map(str, (f1, f2))

    np.save(f1, weights)
    np.save(f2, nodes)

    voxel_model = Array.from_npy(f1, f2)

    assert_array_equal( weights, voxel_model.weights )
    assert_array_equal( nodes, voxel_model.nodes )

# ----------------------------------------------------------------------------
# test
def test_from_fitted_true_array():
    pass

# ============================================================================
# dunder methods
# ----------------------------------------------------------------------------
# test
def test_init():
    # not both arrays
    assert_raises( ValueError, Array, [1], np.array([1]) )

    # wrong sizes
    a, b = map(np.ones, [(10,10), (100,10)])
    assert_raises( ValueError, Array, a, b )

    # dtype mismatch
    b = np.ones((10,10)).astype(np.float32)
    assert_raises( ValueError, Array, a, b )

# ----------------------------------------------------------------------------
# test
def test_getitem(voxel_model, true_array):

    assert_array_equal( voxel_model[3:5,0], true_array[3:5,0] )
    assert_array_equal( voxel_model[2:6], true_array[2:6] )
    assert_array_equal( voxel_model[:,2:6], true_array[:,2:6] )
    assert_array_equal( voxel_model[-1], true_array[-1] )

    idx = [1,3,7]
    assert_array_equal( voxel_model[idx], true_array[idx] )

# ----------------------------------------------------------------------------
# test
def test_len(weights, nodes):
    model = Array(weights, nodes)

    assert( len(weights) == len(model) )

# ============================================================================
# properties
# ----------------------------------------------------------------------------
# test
def test_dtype(weights, nodes):
    model = Array(weights, nodes)

    assert( model.dtype == weights.dtype )
    assert( model.dtype == nodes.dtype )

# ----------------------------------------------------------------------------
# test
def test_shape(voxel_model, true_array):

    assert( voxel_model.shape == true_array.shape )

# ----------------------------------------------------------------------------
# test
def test_size(voxel_model, true_array):

    assert( voxel_model.size == true_array.size )

# ----------------------------------------------------------------------------
# test
def test_T(voxel_model, true_array):

    assert( voxel_model.shape == voxel_model.T.shape[::-1] )

# ============================================================================
# methods
# ----------------------------------------------------------------------------
# test
def test_transpose(voxel_model, true_array):

    assert_array_equal( voxel_model.T[:], true_array.T )

# ----------------------------------------------------------------------------
# test
def test_astype(voxel_model):
    voxel_model = voxel_model.astype(np.float16)

    assert( voxel_model.dtype == np.float16 )
    assert( voxel_model.weights.dtype == np.float16 )
    assert( voxel_model.nodes.dtype == np.float16 )

# ----------------------------------------------------------------------------
# test
def test_sum(voxel_model, true_array):

    assert( voxel_model.sum() == true_array.sum() )
    assert_array_almost_equal( voxel_model.sum(axis=1),
                               true_array.sum(axis=1) )
    assert_array_almost_equal( voxel_model.sum(axis=-1),
                               true_array.sum(axis=-1) )
    assert_array_almost_equal( voxel_model.sum(axis=0),
                               true_array.sum(axis=0) )

    assert_raises( IndexError, voxel_model.sum, axis=2)

# ----------------------------------------------------------------------------
# test
def test_mean(voxel_model, true_array):

    assert( voxel_model.mean() == true_array.mean() )
    assert_array_almost_equal( voxel_model.mean(axis=1),
                               true_array.mean(axis=1) )
    assert_array_almost_equal( voxel_model.mean(axis=-1),
                               true_array.mean(axis=-1) )
    assert_array_almost_equal( voxel_model.mean(axis=0),
                               true_array.mean(axis=0) )

    assert_raises( IndexError, voxel_model.mean, axis=2)

# ----------------------------------------------------------------------------
# test
def test_iterrows(true_array, voxel_model):
    for i, row in enumerate(voxel_model.iterrows()):
        assert_array_equal( row, true_array[i] )

# ----------------------------------------------------------------------------
# test
def test_itercolumns(true_array, voxel_model):
    for j, column in enumerate(voxel_model.itercolumns()):
        assert_array_equal( column, true_array[:, j] )

# ----------------------------------------------------------------------------
# test
def test_iterrows_blocked(true_array, voxel_model):

    rows = np.array_split(np.arange(true_array.shape[0]), 1)
    for i, block in enumerate(voxel_model.iterrows_blocked(n_blocks=1)):
        assert_array_equal( block, true_array[rows[i]] )

    rows = np.array_split(np.arange(true_array.shape[0]), 10)
    for i, block in enumerate(voxel_model.iterrows_blocked(n_blocks=10)):
        assert_array_equal( block, true_array[rows[i]] )

    func = voxel_model.iterrows_blocked(n_blocks=0)
    assert_raises( ValueError, next, func)

    func = voxel_model.iterrows_blocked(n_blocks=20)
    assert_raises( ValueError, next, func)

# ----------------------------------------------------------------------------
# test
def test_itercolumns_blocked(true_array, voxel_model):

    cols = np.array_split(np.arange(true_array.shape[1]), 1)
    for i, block in enumerate(voxel_model.itercolumns_blocked(n_blocks=1)):
        assert_array_equal( block, true_array[:,cols[i]] )

    cols = np.array_split(np.arange(true_array.shape[1]), 10)
    for i, block in enumerate(voxel_model.itercolumns_blocked(n_blocks=10)):
        assert_array_equal( block, true_array[:,cols[i]] )

    func = voxel_model.itercolumns_blocked(n_blocks=0)
    assert_raises( ValueError, next, func)

    func = voxel_model.itercolumns_blocked(n_blocks=20)
    assert_raises( ValueError, next, func)
