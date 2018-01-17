from __future__ import division
import os
import mock
import pytest
import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import csr_matrix

from voxel_model.interpolators import VoxelModel, RegionalizedVoxelModel

@pytest.fixture(scope="module")
def source_voxels():
    return np.argwhere( np.ones((2,2,2)) )

@pytest.fixture(scope="module")
def centroids():
    return np.array([[0, 1, 1],
                     [1, 0, 0],
                     [1, 1, 1]])

@pytest.fixture(scope="module")
def injections():
    # 1 if centroid, else 0
    return np.vstack(
        ( np.array([[[0,0],[0,1]],[[0,0],[0,0]]]).ravel(),
          np.array([[[0,0],[0,0]],[[1,0],[0,0]]]).ravel(),
          np.array([[[0,0],[0,0]],[[0,0],[0,1]]]).ravel() )
    )

@pytest.fixture(scope="module")
def injections_csr(injections):
    return csr_matrix(injections)

@pytest.fixture(scope="module")
def y():
    return np.vstack( (np.ones(8), np.ones(8), np.ones(8)) )

@pytest.fixture(scope="module")
def X(centroids, injections):
    return np.hstack( (centroids, injections) )

@pytest.fixture(scope="module")
def X_csr(centroids, injections):
    return csr_matrix( np.hstack( (centroids, injections) ) )

@pytest.fixture(scope="function")
def voxel_model(source_voxels):
    return VoxelModel(source_voxels, kernel="rbf", gamma=1.0)

@pytest.fixture(scope="module")
def source_key():
    return np.array([[[9,9],[3,3]],[[9,9],[3,3]],[[9,9],[3,3]]]).ravel(),

@pytest.fixture(scope="module")
def target_key():
    return np.array([[[9,9],[3,3]],[[9,9],[3,3]],[[9,9],[3,3]]]).ravel(),

@pytest.fixture(scope="function")
def regionalized_voxel_model(source_voxels, source_key, target_key):
    return RegionalizedVoxelModel(source_voxels, source_key, target_key)

# ----------------------------------------------------------------------------
# test
def test_get_kernel(voxel_model, centroids):
    K = voxel_model._get_kernel(voxel_model.source_voxels, centroids)

    # WHY
    assert( len(K.shape) == 2 )
    # test voxel_model._get_kernel(centroids)

# ----------------------------------------------------------------------------
# test
def test_get_weights(voxel_model, centroids):
    weights = voxel_model._get_weights(centroids)
    n_rows, n_cols = weights.shape

    # correct size
    assert( n_rows == voxel_model.source_voxels.shape[0] )
    assert( n_cols == centroids.shape[1] )

    # assert normalized
    print weights
    assert_array_almost_equal( weights.sum(axis=1), np.ones(n_rows) )

# ----------------------------------------------------------------------------
# test
def test_voxel_fit(voxel_model, X, X_csr, y):
    dense = voxel_model.fit(X, y)
    sparse = voxel_model.fit(X_csr, y)

    assert_array_almost_equal( dense.weights_, sparse.weights_ )

# ----------------------------------------------------------------------------
# test
def test_voxel_predict_dense(voxel_model, centroids, X, X_csr, y):
    dense = voxel_model.fit(X, y).predict(X)

    # model @ centroids should be close to equal???
    assert_array_almost_equal( dense, y )

# ----------------------------------------------------------------------------
# test
def test_get_voxel_matrix(voxel_model):
    pass

# ----------------------------------------------------------------------------
# test
def test_region_fit(regionalized_voxel_model):
    pass

# ----------------------------------------------------------------------------
# test
def test_region_predict(regionalized_voxel_model):
    pass

# ----------------------------------------------------------------------------
# test
def test_get_region_matrix(regionalized_voxel_model):
    pass
