from __future__ import division
import os
import mock
import pytest
import numpy as np

from numpy.testing import assert_array_equal, assert_raises

from voxel_model.core import VoxelModelData
from voxel_model.tests.conftest import mcc, tree, annotation

# =============================================================================
# VoxelModelData class
# -----------------------------------------------------------------------------
# tests
def test_default_structure_ids():
    # TODO not yet tested in sdk, assume solid for now
    pass


# -----------------------------------------------------------------------------
# tests
def test_experiment_generator(mcc):
    def get_experiment_list(data):
        return list(data._experiment_generator(experiment_ids))
    experiment_ids = [456, 12]

    # full
    data = VoxelModelData(mcc)
    exps = get_experiment_list(data)

    assert len(exps) == 2

    # min inj/proj
    data = VoxelModelData(mcc, min_injection_sum=np.inf)
    exps = get_experiment_list(data)

    assert not exps

    # hemisphere id
    data = VoxelModelData(mcc, injection_hemisphere_id=0)
    exps = get_experiment_list(data)

    assert not exps

    # TODO: flip option
    # data = _BaseModelData(mcc, injection_hemisphere_id=1, flip_experiments=True)


# -----------------------------------------------------------------------------
# tests
def test_get_experiment_data(mcc):
    experiment_ids = [456, 12]

    # mock masks
    mask = mock.Mock()
    mask.mask_volume.side_effect = lambda x: x

    data = VoxelModelData(mcc)
    data.get_experiment_data(experiment_ids)

    assert data.centroids.shape == (2, 3)
    assert data.injections.shape[0] == 2
    assert data.projections.shape[0] == 2
    assert data.injections.shape[1] == data.projections.shape[1]
