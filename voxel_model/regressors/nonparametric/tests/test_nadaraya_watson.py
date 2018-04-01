import pytest
import numpy as np

from scipy.sparse import csr_matrix
from numpy.testing \
    import assert_array_equal, assert_array_almost_equal, assert_raises

from voxel_model.regressors import NadarayaWatson

@pytest.fixture(scope="module")
def centroids():
    return np.random.rand(5,3)

@pytest.fixture(scope="module")
def y():
    return np.random.rand(5,200)

@pytest.fixture(scope="module")
def source_voxels():
    return np.random.rand(100, 3)

@pytest.fixture(scope="function")
def model():
    return NadarayaWatson(kernel="rbf", gamma=1.0)

# ----------------------------------------------------------------------------
# test
def test_get_kernel(model, centroids, y):
    pass

# ----------------------------------------------------------------------------
# test
def test_compute_weights():
    pass

# ----------------------------------------------------------------------------
# test
def test_check_fit_arrays():
    pass

# ----------------------------------------------------------------------------
# test
def test_predict(model, source_voxels, centroids, y):
    dense = model.fit(centroids, y).predict(source_voxels)
    sparse_y = model.fit(centroids, csr_matrix(y)).predict(source_voxels)
    sparse_x = model.fit(csr_matrix(centroids), y).predict(source_voxels)
    sparse_p = model.fit(centroids, y).predict(csr_matrix(source_voxels))


    assert_array_almost_equal( dense, sparse_y )
    assert_array_almost_equal( dense, sparse_x )
    assert_array_almost_equal( dense, sparse_p )

# ----------------------------------------------------------------------------
# test
def test_get_weights():
    pass
