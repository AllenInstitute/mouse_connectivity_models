from __future__ import division
import pytest
import numpy as np

from numpy.testing \
    import assert_array_equal, assert_array_almost_equal, assert_raises

from voxel_model.core.masks import Mask
from voxel_model.core.experiment \
    import (_pull_grid_data, _mask_data_volume,
            _compute_true_injection_density, Experiment)
from voxel_model.core.tests.conftest import mcc


# ============================================================================
# Module Level Functions
# ============================================================================
def test_pull_grid_data(mcc):
    # ------------------------------------------------------------------------
    # tests pull 'data' from mcc fixture
    experiment_id = 1023223
    data_volumes = _pull_grid_data(mcc, experiment_id)

    assert all([isinstance(x, np.ndarray) for x in data_volumes.values()])


def test_mask_data_volume():
    # ------------------------------------------------------------------------
    # tests mask data volume with thresholds
    a = np.ones((4, 4))
    mask = np.ones((4, 4))
    mask[0:2] = 0.3

    assert_array_equal(_mask_data_volume(a, mask, 0.1), np.ones((4, 4)))
    assert_array_equal(_mask_data_volume(a, mask, 0.5)[0:2], np.zeros((2, 4)))

    # ------------------------------------------------------------------------
    # test inplace
    assert_array_equal(a[0:2], np.zeros((2, 4)))

    # ------------------------------------------------------------------------
    # test incompatible data volume shape
    assert_raises(ValueError, _mask_data_volume, a, np.ones((3, 3)))


def test_compute_true_injection_density():
    # ------------------------------------------------------------------------
    # test ouput is correct
    a = np.ones((4, 4))
    b = np.zeros((4, 4))

    assert_array_equal(_compute_true_injection_density(a, b), b)
    assert_array_equal(_compute_true_injection_density(a, b, inplace=True), b)

    # ------------------------------------------------------------------------
    # test inplace
    assert_array_equal(a, b)

    # ------------------------------------------------------------------------
    # test incompatible shapes
    assert_raises(ValueError, _compute_true_injection_density, a,
                  np.zeros((3, 3)))


# ============================================================================
# Experiment Class
# ============================================================================
def test_from_mcc(mcc):

    # ------------------------------------------------------------------------
    # test correct data volumes pulled
    experiment_id = 1023223
    injection_density = mcc.get_injection_density(experiment_id)[0]
    injection_fraction = mcc.get_injection_fraction(experiment_id)[0]
    projection_density = mcc.get_projection_density(experiment_id)[0]

    # 'true' injection density
    _compute_true_injection_density(injection_density, injection_fraction,
                                    inplace=True)

    experiment = Experiment.from_mcc(mcc, experiment_id)
    assert_array_equal(experiment.injection_density, injection_density)
    assert_array_equal(experiment.projection_density, projection_density)


def test_get_injection():
    # ------------------------------------------------------------------------
    # test correct injection density returned
    injd = np.random.rand(27).reshape(3, 3, 3)
    prjd = np.random.rand(27).reshape(3, 3, 3)

    norm_injd = injd / injd.sum()

    experiment = Experiment(injd, prjd)

    assert_array_equal(experiment.get_injection(), injd)
    assert_array_equal(experiment.get_injection(True), norm_injd)


def test_get_projection_density():
    # ------------------------------------------------------------------------
    # test correct projection density returned
    injd = np.random.rand(27).reshape(3, 3, 3)
    prjd = np.random.rand(27).reshape(3, 3, 3)

    norm_prjd = prjd / injd.sum()

    experiment = Experiment(injd, prjd)

    assert_array_equal(experiment.get_projection(), prjd)
    assert_array_equal(experiment.get_projection(True), norm_prjd)


def test_flip():
    # ------------------------------------------------------------------------
    # test data volumes are flipped along midline (last axis)
    ones, zeros = np.ones((4, 4)), np.zeros((4, 4))
    left, right = np.dstack((ones, zeros)), np.dstack((zeros, ones))

    experiment = Experiment(left, right)
    experiment.flip()

    assert_array_equal(experiment.injection_density, right)
    assert_array_equal(experiment.projection_density, left)
