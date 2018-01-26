from __future__ import division
import os
import mock
import pytest
import numpy as np

from numpy.testing import assert_array_equal, assert_raises
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi

from voxel_model.experiment import Experiment

# -----------------------------------------------------------------------------
# tests
def test_from_mcc(mcc):

    # pull 'data' from mcc fixture
    experiment_id = 1023223
    injection_density = mcc.get_injection_density(experiment_id)[0]
    injection_fraction = mcc.get_injection_fraction(experiment_id)[0]
    projection_density = mcc.get_projection_density(experiment_id)[0]

    # 'true' injection density from constructor
    np.multiply(injection_density, injection_fraction, injection_density)

    experiment = Experiment.from_mcc(mcc, experiment_id)

    assert_array_equal( experiment.injection_density, injection_density )
    assert_array_equal( experiment.projection_density, projection_density )

# -----------------------------------------------------------------------------
# tests
def test_get_injection_hemisphere(mcc):
    # pull 'data' from mcc fixture
    experiment_id = 1023223
    injection_density = mcc.get_injection_density(experiment_id)[0]
    projection_density = mcc.get_projection_density(experiment_id)[0]

    # left injection
    l_inj = np.copy(injection_density)
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
def test_sum_injection(mcc):
    # pull 'data' from mcc fixture
    experiment_id = 1023223
    injection_density = mcc.get_injection_density(experiment_id)[0]
    injection_fraction = mcc.get_injection_fraction(experiment_id)[0]

    # 'true' injection density from constructor
    np.multiply(injection_density, injection_fraction, injection_density)

    experiment = Experiment.from_mcc(mcc, experiment_id)

    assert( experiment.injection_density.sum() == injection_density.sum() )

# -----------------------------------------------------------------------------
# tests
def test_compute_centroid(mcc):
    # pull 'data' from mcc fixture
    experiment_id = 1023223
    injection_density = mcc.get_injection_density(experiment_id)[0]
    injection_fraction = mcc.get_injection_fraction(experiment_id)[0]

    # compute allensdk centroid
    api = MouseConnectivityApi()
    mcc_centroid = api.calculate_injection_centroid( injection_density,
                                                     injection_fraction,
                                                     resolution=1 )

    experiment = Experiment.from_mcc(mcc, experiment_id)

    assert_array_equal( experiment.centroid, mcc_centroid )

# -----------------------------------------------------------------------------
# tests
def test_normalized_injection_density(mcc):

    experiment_id = 1023223
    experiment = Experiment.from_mcc(mcc, experiment_id)

    assert( experiment.normalized_injection_density.dtype == np.float )

# -----------------------------------------------------------------------------
# tests
def test_normalized_projection_density(mcc):

    experiment_id = 1023223
    experiment = Experiment.from_mcc(mcc, experiment_id)

    assert( experiment.normalized_projection_density.dtype == np.float )

