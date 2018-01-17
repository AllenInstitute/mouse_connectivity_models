from __future__ import division
import os
import mock
import pytest
import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal
from voxel_model.experiment import Experiment

@pytest.fixture(scope="module")
def data_mask():
    return np.array([[[1,1],[0.8,1]],[[0.8,1],[0.2,0.8]]])

@pytest.fixture(scope="function")
def experiment(data_mask):
    mcc = mock.Mock()
    mcc.get_data_mask.return_value = ( data_mask, )
    mcc.get_injection_density.return_value = ( np.random.rand(2,2,2), )
    mcc.get_projection_density.return_value = ( np.random.rand(2,2,2), )

    return Experiment(mcc, 1013)

# ----------------------------------------------------------------------------
# test
def test_mask_to_valid(experiment):
    test = np.array([[[1,1],[1,1]],[[1,1],[0,1]]])
    masked = experiment._mask_to_valid( np.ones((2,2,2)) )

    assert_array_equal( masked, test )

# ----------------------------------------------------------------------------
# test
def test_projection_density(experiment):
    # is it actually normalized
    prd = experiment.projection_density
    inj = experiment.injection_density

    experiment.normalize_projection = True
    normalized_prd = experiment.projection_density

    assert_array_almost_equal( prd, normalized_prd*inj.sum() )
