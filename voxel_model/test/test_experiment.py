from __future__ import division
import os
import mock
import pytest
import numpy as np

from numpy.testing import assert_array_equal, assert_raises
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi

from voxel_model.experiment import Experiment

@pytest.fixture(scope="module")
def shape():
    return (10,10,10)

@pytest.fixture(scope="module")
def data_mask(shape):
    return np.ones(shape)

@pytest.fixture(scope="module")
def injection_density(shape):
    return np.ones(shape)

@pytest.fixture(scope="module")
def injection_fraction(injection_density):
    return injection_density != 0

@pytest.fixture(scope="module")
def projection_density(shape):
    return np.ones(shape)

@pytest.fixture(scope="module")
def mcc(data_mask, injection_density, injection_fraction, projection_density):

    mcc = mock.Mock()

    mcc.get_data_mask.return_value = (data_mask, )
    mcc.get_injection_density.return_value = (injection_density, )
    mcc.get_injection_fraction.return_value = (injection_fraction, )
    mcc.get_projection_density.return_value = (projection_density, )

    mcc.api.return_value = MouseConnectivityApi()

    return mcc

# -----------------------------------------------------------------------------
# tests
def test_from_mcc(mcc, injection_density, injection_fraction, projection_density):
    experiment_id = 1023223

    data = Experiment.from_mcc(mcc, experiment_id)

    # 'true' injection density from constructor
    np.multiply(injection_density, injection_fraction, injection_density)

    assert_array_equal( data.injection_density, injection_density )
    assert_array_equal( data.projection_density, projection_density )

# -----------------------------------------------------------------------------
# tests
def test_check_injection_hemisphere(injection_density, projection_density):

    # left injection
    l_inj = np.copy(injectiÂon_density)
    l_inj[...,l_inj.shape[-1]//2:] = 0
    l_data = Experiment(l_inj, projection_density)

    # should be flipped
    assert_array_equal( l_inj, l_data.injection_density[...,::-1] )

    # right injection
    r_inj = np.copy(injection_density)
    r_inj[...,:r_inj.shape[-1]//2] = 0
    r_data = Experiment(r_inj, projection_density)

    # should not be flipped
    assert_array_equal( r_inj, r_data.injection_density )


# -----------------------------------------------------------------------------
# tests
def test_sum_injection(mcc, injection_density, injection_fraction):
    experiment_id = 1023223

    data = Experiment.from_mcc(mcc, experiment_id)

    # 'true' injection density from constructor
    np.multiply(injection_density, injection_fraction, injection_density)

    assert( data.sum_injection == injection_density.sum() )

# -----------------------------------------------------------------------------
# tests
def test_centroid(mcc, injection_density, injection_fraction):

    experiment_id = 1023223
    data = Experiment.from_mcc(mcc, experiment_id)

    api = MouseConnectivityApi()
    mcc_centroid = api.calculate_injection_centroid( injection_density,
                                                     injection_fraction,
                                                     resolution=1 )

    assert_array_equal( data.centroid, mcc_centroid )

# -----------------------------------------------------------------------------
# tests
def test_normalized_injection_density(mcc):

    experiment_id = 1023223
    data = Experiment.from_mcc(mcc, experiment_id)

    assert( data.normalized_injection_density.dtype == np.float )

# -----------------------------------------------------------------------------
# tests
def test_normalized_projection_density(mcc):

    experiment_id = 1023223
    data = Experiment.from_mcc(mcc, experiment_id)

    assert( data.normalized_projection_density.dtype == np.float )

