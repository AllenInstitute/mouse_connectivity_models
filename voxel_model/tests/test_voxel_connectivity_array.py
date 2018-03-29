"""
NOTE :: NEED to add testing for __init__ esp for passing fitted true_array
"""
from __future__ import division
import pytest
import numpy as np
from numpy.testing \
    import assert_array_equal, assert_array_almost_equal, assert_raises

from voxel_model.voxel_connectivity_array import VoxelConnectivityArray

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
def voxel_array(weights, nodes):
    return VoxelConnectivityArray(weights=weights, nodes=nodes)

# ============================================================================
# constructors
# ----------------------------------------------------------------------------
# test
def test_from_hdf5():
    args = ("weights.hdf5", "nodes.hdf5")
    assert_raises( NotImplementedError, VoxelConnectivityArray.from_hdf5, *args )

# ----------------------------------------------------------------------------
# test
def test_from_csv(tmpdir, weights, nodes):
    f1 = tmpdir.join("weights.csv")
    f2 = tmpdir.join("nodes.csv")

    # get filenames not localpath objects
    f1, f2 = map(str, (f1, f2))

    np.savetxt(f1, weights, delimiter=",")
    np.savetxt(f2, nodes, delimiter=",")

    voxel_array = VoxelConnectivityArray.from_csv(f1, f2)

    assert_array_equal( weights, voxel_array.weights )
    assert_array_equal( nodes, voxel_array.nodes )

# ----------------------------------------------------------------------------
# test
def test_from_npy(tmpdir, weights, nodes):
    f1 = tmpdir.join("weights.npy")
    f2 = tmpdir.join("nodes.npy")

    # get filenames not localpath objects
    f1, f2 = map(str, (f1, f2))

    np.save(f1, weights)
    np.save(f2, nodes)

    voxel_array = VoxelConnectivityArray.from_npy(f1, f2)

    assert_array_equal( weights, voxel_array.weights )
    assert_array_equal( nodes, voxel_array.nodes )

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
    assert_raises( AttributeError, VoxelConnectivityArray, [1], np.array([1]) )

    # wrong sizes
    a, b = map(np.ones, [(10,10), (100,10)])
    assert_raises( ValueError, VoxelConnectivityArray, a, b )

    # dtype mismatch
    b = np.ones((10,10)).astype(np.float32)
    assert_raises( ValueError, VoxelConnectivityArray, a, b )

# ----------------------------------------------------------------------------
# test
def test_getitem(voxel_array, true_array):

    assert_array_equal( voxel_array[3:5,0], true_array[3:5,0] )
    assert_array_equal( voxel_array[2:6], true_array[2:6] )
    assert_array_equal( voxel_array[:,2:6], true_array[:,2:6] )
    assert_array_equal( voxel_array[-1], true_array[-1] )

    idx = [1,3,7]
    assert_array_equal( voxel_array[idx], true_array[idx] )

# ----------------------------------------------------------------------------
# test
def test_len(weights, nodes):
    model = VoxelConnectivityArray(weights, nodes)

    assert( len(weights) == len(model) )

# ============================================================================
# properties
# ----------------------------------------------------------------------------
# test
def test_dtype(weights, nodes):
    model = VoxelConnectivityArray(weights, nodes)

    assert( model.dtype == weights.dtype )
    assert( model.dtype == nodes.dtype )

# ----------------------------------------------------------------------------
# test
def test_shape(voxel_array, true_array):

    assert( voxel_array.shape == true_array.shape )

# ----------------------------------------------------------------------------
# test
def test_size(voxel_array, true_array):

    assert( voxel_array.size == true_array.size )

# ----------------------------------------------------------------------------
# test
def test_T(voxel_array, true_array):

    assert( voxel_array.shape == voxel_array.T.shape[::-1] )

# ============================================================================
# methods
# ----------------------------------------------------------------------------
# test
def test_transpose(voxel_array, true_array):

    assert_array_equal( voxel_array.T[:], true_array.T )

# ----------------------------------------------------------------------------
# test
def test_astype(voxel_array):
    voxel_array = voxel_array.astype(np.float16)

    assert( voxel_array.dtype == np.float16 )
    assert( voxel_array.weights.dtype == np.float16 )
    assert( voxel_array.nodes.dtype == np.float16 )

# ----------------------------------------------------------------------------
# test
def test_sum(voxel_array, true_array):

    assert( voxel_array.sum() == true_array.sum() )
    assert_array_almost_equal( voxel_array.sum(axis=1),
                               true_array.sum(axis=1) )
    assert_array_almost_equal( voxel_array.sum(axis=-1),
                               true_array.sum(axis=-1) )
    assert_array_almost_equal( voxel_array.sum(axis=0),
                               true_array.sum(axis=0) )

    assert_raises( IndexError, voxel_array.sum, axis=2)

# ----------------------------------------------------------------------------
# test
def test_mean(voxel_array, true_array):

    assert( voxel_array.mean() == true_array.mean() )
    assert_array_almost_equal( voxel_array.mean(axis=1),
                               true_array.mean(axis=1) )
    assert_array_almost_equal( voxel_array.mean(axis=-1),
                               true_array.mean(axis=-1) )
    assert_array_almost_equal( voxel_array.mean(axis=0),
                               true_array.mean(axis=0) )

    assert_raises( IndexError, voxel_array.mean, axis=2)

# ----------------------------------------------------------------------------
# test
def test_iterrows(true_array, voxel_array):
    for i, row in enumerate(voxel_array.iterrows()):
        assert_array_equal( row, true_array[i] )

# ----------------------------------------------------------------------------
# test
def test_itercolumns(true_array, voxel_array):
    for j, column in enumerate(voxel_array.itercolumns()):
        assert_array_equal( column, true_array[:, j] )

# ----------------------------------------------------------------------------
# test
def test_iterrows_blocked(true_array, voxel_array):

    rows = np.array_split(np.arange(true_array.shape[0]), 1)
    for i, block in enumerate(voxel_array.iterrows_blocked(n_blocks=1)):
        assert_array_equal( block, true_array[rows[i]] )

    rows = np.array_split(np.arange(true_array.shape[0]), 10)
    for i, block in enumerate(voxel_array.iterrows_blocked(n_blocks=10)):
        assert_array_equal( block, true_array[rows[i]] )

    func = voxel_array.iterrows_blocked(n_blocks=0)
    assert_raises( ValueError, next, func)

    func = voxel_array.iterrows_blocked(n_blocks=20)
    assert_raises( ValueError, next, func)

# ----------------------------------------------------------------------------
# test
def test_itercolumns_blocked(true_array, voxel_array):

    cols = np.array_split(np.arange(true_array.shape[1]), 1)
    for i, block in enumerate(voxel_array.itercolumns_blocked(n_blocks=1)):
        assert_array_equal( block, true_array[:,cols[i]] )

    cols = np.array_split(np.arange(true_array.shape[1]), 10)
    for i, block in enumerate(voxel_array.itercolumns_blocked(n_blocks=10)):
        assert_array_equal( block, true_array[:,cols[i]] )

    func = voxel_array.itercolumns_blocked(n_blocks=0)
    assert_raises( ValueError, next, func)

    func = voxel_array.itercolumns_blocked(n_blocks=20)
    assert_raises( ValueError, next, func)
