import pytest
import numpy as np

from scipy.sparse import csr_matrix, csc_matrix
from numpy.testing \
    import assert_array_equal, assert_array_almost_equal, assert_raises

from mcmodels.models.voxel import VoxelModel

@pytest.fixture(scope="module")
def centroids():
    return np.random.rand(5,3)

@pytest.fixture(scope="module")
def injections():
    # 1 if centroid, else 0
    return np.random.rand(5,100)

@pytest.fixture(scope="module")
def y():
    return np.random.rand(5,200)

@pytest.fixture(scope="module")
def X(centroids, injections):
    return np.hstack( (centroids, injections) )

@pytest.fixture(scope="function")
def model():
    return VoxelModel(np.random.rand(100,3), kernel="rbf", gamma=1.0)

# ----------------------------------------------------------------------------
# test
def test_fit_sparse_centroids(model, centroids, injections, X, y):
    dense = model.fit((centroids, injections), y)
    sparse_csr = model.fit((csr_matrix(centroids), injections), y)
    sparse_csc = model.fit((csc_matrix(centroids), injections), y)

    assert_array_almost_equal( dense.weights_, sparse_csr.weights_ )
    assert_array_almost_equal( dense.weights_, sparse_csc.weights_ )

# ----------------------------------------------------------------------------
# test
def test_fit_sparse_injections(model, centroids, injections, X, y):
    dense = model.fit((centroids, injections), y)
    sparse_csr = model.fit((centroids, csr_matrix(injections)), y)
    sparse_csc = model.fit((centroids, csc_matrix(injections)), y)

    assert_array_almost_equal( dense.weights_, sparse_csr.weights_ )
    assert_array_almost_equal( dense.weights_, sparse_csc.weights_ )

# ----------------------------------------------------------------------------
# test
def test_fit_sparse_y(model, centroids, injections, X, y):
    dense = model.fit((centroids, injections), y)
    sparse_csr = model.fit((centroids, injections), csr_matrix(y) )
    sparse_csc = model.fit((centroids, injections), csc_matrix(y) )

    assert_array_almost_equal( dense.weights_, sparse_csr.weights_ )
    assert_array_almost_equal( dense.weights_, sparse_csc.weights_ )

# ----------------------------------------------------------------------------
# test
def test_predict(model, centroids, X, y):
    dense = model.fit(X, y).predict(X)
    sparse_csr = model.fit(X, y).predict(csr_matrix(X))
    sparse_csc = model.fit(X, y).predict(csc_matrix(X))

    assert_array_almost_equal( dense, sparse_csr )
    assert_array_almost_equal( dense, sparse_csc )

