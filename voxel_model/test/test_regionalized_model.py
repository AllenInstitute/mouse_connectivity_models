from __future__ import division
import os
import mock
import pytest
import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.sparse import csr_matrix

from voxel_model.regionalized_model \
    import RegionalizedModel, _generate_column_sets

@pytest.fixture(scope="module")
def source_key():
    return np.array([[[9,9],[3,3]],[[9,9],[3,3]],[[9,9],[3,3]]]).ravel(),

@pytest.fixture(scope="module")
def target_key():
    return np.array([[[9,9],[3,3]],[[9,9],[3,3]],[[9,9],[3,3]]]).ravel(),

# ============================================================================
# Module level functions
# ----------------------------------------------------------------------------
# test
def test_generate_column_sets():
    pass

# ============================================================================
# RegionalizedModel class
# ----------------------------------------------------------------------------
# test
def test_predict():
    pass

# ----------------------------------------------------------------------------
# test
def test_get_unique_counts():
    pass

# ----------------------------------------------------------------------------
# test
def test_get_region_matrix():
    pass

# ----------------------------------------------------------------------------
# test
def test_get_metric():
    pass
