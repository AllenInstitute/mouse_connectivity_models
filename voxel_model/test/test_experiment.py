from __future__ import division
import os
import mock
import pytest
import numpy as np

from numpy.testing import assert_array_equal, assert_raises
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi

from voxel_model.masks import Mask

from voxel_model.experiment \
    import ( _pull_data_volumes, _mask_data_volume,
             _compute_true_injection_density, _get_injection_hemisphere,
             _flip_hemisphere, compute_centroid, Experiment )

@pytest.fixture(scope="module")
def experiment(mcc):
    experiment_id=1223452 # whatever
    return Experiment.from_mcc(mcc, experiment_id)

@pytest.fixture(scope="module")
def mask(mcc):
    structure_ids=[2,3]
    return Mask(mcc, structure_ids, hemisphere=2)

# =============================================================================
# Module Level Functions
# -----------------------------------------------------------------------------
# tests
def test_pull_data_volumes(mcc):
    # pull 'data' from mcc fixture
    experiment_id = 1023223
    data_volumes = _pull_data_volumes(mcc, experiment_id)

    assert( all([ isinstance(x, np.ndarray) for x in data_volumes.values() ]) )

# -----------------------------------------------------------------------------
# tests
def test_mask_data_volume(mcc):
    pass

# -----------------------------------------------------------------------------
# tests
def test_compute_true_injection_density(mcc):
    pass

# -----------------------------------------------------------------------------
# tests
def test_get_injection_hemisphere(mcc):
    pass

# -----------------------------------------------------------------------------
# tests
def test_flip_hemipshere(mcc):
    pass
#    # pull 'data' from mcc fixture
#    experiment_id = 1023223
#    injection_density = mcc.get_injection_density(experiment_id)[0]
#    projection_density = mcc.get_projection_density(experiment_id)[0]
#
#    # left injection
#    l_inj = np.copy(injection_density)
#    l_inj[...,l_inj.shape[-1]//2:] = 0
#    l_data = Experiment(l_inj, projection_density)
#
#    # should be flipped
#    assert_array_equal( l_inj, l_data.injection_density[...,::-1] )
#
#    # right injection
#    r_inj = np.copy(injection_density)
#    r_inj[...,:r_inj.shape[-1]//2] = 0
#    r_data = Experiment(r_inj, projection_density)
#
#    # should not be flipped
#    assert_array_equal( r_inj, r_data.injection_density )

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

    # 'true' injection density
    _compute_true_injection_density( injection_density, injection_fraction,
                                     inplace=True )

    assert_array_equal( compute_centroid(injection_density) , mcc_centroid )

# =============================================================================
# Experiment Class
# -----------------------------------------------------------------------------
# tests
def test_from_mcc(mcc, experiment):

    # pull 'data' from mcc fixture
    experiment_id = 1023223
    injection_density = mcc.get_injection_density(experiment_id)[0]
    injection_fraction = mcc.get_injection_fraction(experiment_id)[0]
    projection_density = mcc.get_projection_density(experiment_id)[0]

    # 'true' injection density
    _compute_true_injection_density( injection_density, injection_fraction,
                                     inplace=True )

    assert_array_equal( experiment.injection_density, injection_density )
    assert_array_equal( experiment.projection_density, projection_density )

# -----------------------------------------------------------------------------
# tests
def test_normalized_injection_density(experiment):

    assert( experiment.normalized_injection_density.dtype == np.float )

# -----------------------------------------------------------------------------
# tests
def test_normalized_projection_density(experiment):

    assert( experiment.normalized_projection_density.dtype == np.float )

# -----------------------------------------------------------------------------
# tests
def get_injection_ratio_contained(experiment, mask):
    pass

# -----------------------------------------------------------------------------
# tests
def mask_volume(experiment, mask):
    pass
