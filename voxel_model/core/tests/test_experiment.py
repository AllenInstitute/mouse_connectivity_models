from __future__ import division
import os
import mock
import pytest
import numpy as np

from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
from numpy.testing \
    import assert_array_equal, assert_array_almost_equal, assert_raises

from voxel_model.masks import Mask
from voxel_model.experiment \
    import ( _pull_data_volumes, _mask_data_volume,
             _compute_true_injection_density, _get_injection_hemisphere,
             _flip_hemisphere, compute_centroid, Experiment )

@pytest.fixture(scope="module")
def experiment(mcc):
    experiment_id=1223452 # whatever
    return Experiment.from_mcc(mcc, experiment_id)

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
def test_mask_data_volume():
    a = np.ones((4,4))
    mask = np.ones((4,4))
    mask[0:2] = 0.3

    assert_array_equal( _mask_data_volume(a, mask, 0.1), np.ones((4,4)) )
    assert_array_equal( _mask_data_volume(a, mask, 0.5)[0:2], np.zeros((2,4)) )

    # test inplace
    assert_array_equal( a[0:2], np.zeros((2,4)) )

    assert_raises( ValueError, _mask_data_volume, a, np.ones((3,3)) )

# -----------------------------------------------------------------------------
# tests
def test_compute_true_injection_density():
    a = np.ones((4,4))
    b = np.zeros((4,4))

    assert_array_equal( _compute_true_injection_density(a, b), b )
    assert_array_equal( _compute_true_injection_density(a, b, inplace=True), b)

    # test inplace
    assert_array_equal( a, b )

    assert_raises( ValueError, _compute_true_injection_density, a,
                   np.zeros((3,3)) )
# -----------------------------------------------------------------------------
# tests
def test_get_injection_hemisphere():
    o, z = np.ones((4,4)), np.zeros((4,4))
    l, r = np.dstack( (o,z) ), np.dstack( (z,o) )

    assert( _get_injection_hemisphere(l) == 1 )
    assert( _get_injection_hemisphere(r) == 2 )

    assert_raises( ValueError, _get_injection_hemisphere, np.ones((4,4)) )


# -----------------------------------------------------------------------------
# tests
def test_flip_hemipshere(mcc):
    o, z = np.ones((4,4)), np.zeros((4,4))
    l, r = np.dstack( (o,z) ), np.dstack( (z,o) )

    assert_array_equal( _flip_hemisphere(l), r )
    assert_array_equal( _flip_hemisphere(r), l )

    assert_raises( ValueError, _flip_hemisphere, np.ones((4,4)) )

# -----------------------------------------------------------------------------
# tests
def test_compute_centroid():
    # pull 'data' from mcc fixture
    a = np.random.rand(4,4,4)
    b = np.random.rand(4,4,4)

    # compute allensdk centroid
    api = MouseConnectivityApi()
    mcc_centroid = api.calculate_injection_centroid( a, b, 1)

    # 'true' injection density
    _compute_true_injection_density( a, b, inplace=True )

    assert_array_almost_equal( compute_centroid(a) , mcc_centroid )

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

    # invalid injection hemisphere
    assert_raises( ValueError, Experiment.from_mcc, mcc, experiment_id, 4 )

# -----------------------------------------------------------------------------
# tests
def test_normalized_injection_density(experiment):

    assert( experiment.normalized_injection_density.shape == \
            experiment.injection_density.shape )
    assert( experiment.normalized_injection_density.dtype == np.float )

# -----------------------------------------------------------------------------
# tests
def test_normalized_projection_density(experiment):

    assert( experiment.normalized_projection_density.shape == \
            experiment.projection_density.shape )
    assert( experiment.normalized_projection_density.dtype == np.float )

# -----------------------------------------------------------------------------
# tests
def get_injection_ratio_contained(experiment):
    # np.ndarray
    mask = np.ones_like(experiment.injection_density)
    mask[..., :mask.shape[2]//2] = 0

    assert( experiment.get_injection_ratio_contained(mask) == 0.5 )

    # Mask object
    mask = Mask(mcc, [2,3], hemisphere=3)
    assert( type(experiment.get_injection_ratio_contained(mask)) == float )

    # wrong np.ndarray size
    assert_raises(ValueError, experiment.get_injection_density, np.ones((2,2)))

# -----------------------------------------------------------------------------
# tests
def mask_volume(experiment):
    # np.ndarray
    mask = np.ones_like(experiment.injection_density)
    mask[..., :mask.shape[2]//2] = 0

    n_nnz = len(mask.nonzero()[0])
    assert(experiment.mask_volume("injection_density", mask).shape == (n_nnz,))

    # Mask object
    mask = Mask(mcc, [2,3], hemisphere=3)
    assert( len(experiment.mask_volume("injection_density", mask).shape) == 1  )

    # wrong np.ndarray size
    assert_raises( ValueError, experiment.mask_volume, "data", mask )
    assert_raises( ValueError, experiment.mask_volume, "injection_density",
                   np.ones((2,2)) )
