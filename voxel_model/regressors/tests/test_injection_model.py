from __future__ import division
import os
import mock
import pytest
import numpy as np

from scipy.sparse import csr_matrix
from numpy.testing \
    import assert_array_equal, assert_array_almost_equal, assert_raises

from voxel_model.regressors import InjectionModel

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
    return InjectionModel(np.random.rand(100,3), kernel="rbf", gamma=1.0)

# ----------------------------------------------------------------------------
# test
def test_fit(model, centroids, injections, X, y):
    dense = model.fit((centroids, injections), y)
    sparse = model.fit((centroids, csr_matrix(injections)), y)

    assert_array_almost_equal( dense.weights_, sparse.weights_ )

# ----------------------------------------------------------------------------
# test
def test_predict_dense(model, centroids, X, y):
    dense = model.fit(X, y).predict(X)
    sparse = model.fit(csr_matrix(X), y).predict(X)

    # model @ centroids should be close to equal???
    assert_array_almost_equal( dense, sparse )
