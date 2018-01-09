from __future__ import division
import os
import mock
import pytest
import numpy as np

from voxel_model.interpolators import VoxelModel, RegionalizedVoxelModel

@pytest.fixture(scope="function")
def voxel_model():
    pass

@pytest.fixture(scope="function")
def regionalized_voxel_model():
    pass

# ----------------------------------------------------------------------------
# test 
def test_get_kernel(voxel_model):
    pass

# ----------------------------------------------------------------------------
# test 
def test_get_weights(voxel_model):
    pass

# ----------------------------------------------------------------------------
# test 
def test_voxel_fit(voxel_model):
    pass

# ----------------------------------------------------------------------------
# test 
def test_voxel_predict(voxel_model):
    pass

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
