from __future__ import division
import os
import mock
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
def test_predict(model, source_voxels, centroids, y):
    dense = model.fit(centroids, y).predict(source_voxels)
    sparse = model.fit(centroids, csr_matrix(y)).predict(source_voxels)


    assert_array_almost_equal( dense, sparse )
