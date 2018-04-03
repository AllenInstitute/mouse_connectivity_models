from __future__ import division
import os
import mock
import pytest
import numpy as np

from numpy.testing import assert_array_equal, assert_raises

from voxel_model.core import VoxelModelData
from voxel_model.core.tests.conftest import mcc, mask

@pytest.fixture(scope="function")
def Data():
    data = VoxelModelData
    data.default_structure_ids = [6, 7]
    return data


# =============================================================================
# VoxelModelData class
# =============================================================================
def test_default_structure_ids():
    # TODO not yet tested in sdk, assume solid for now
    pass


def test_experiment_generator(mcc, Data):
    def get_experiment_size(data):
        size = 0
        for _ in data._experiment_generator(experiment_ids):
            size += 1
        return size
    # ------------------------------------------------------------------------
    # tests pulls all with no option
    experiment_ids = [456, 12]

    # full
    data = Data(mcc)
    exps = get_experiment_size(data)

    assert exps == 2

    # ------------------------------------------------------------------------
    # tests pulls none when min_inj_sum too high
    data = Data(mcc, min_injection_sum=np.inf)
    exps = get_experiment_size(data)

    assert exps == 0

    # ------------------------------------------------------------------------
    # tests pulls none when hemi id is invalid
    data = Data(mcc, injection_hemisphere_id=-5, flip_experiments=False)
    exps = get_experiment_size(data)

    assert exps == 0

    # TODO: flip option
    # data = _BaseModelData(mcc, injection_hemisphere_id=1, flip_experiments=True)


def test_get_experiment_data(Data, mcc, mask):
    # ------------------------------------------------------------------------
    # tests data arrays are set as attrs and have proper shape
    experiment_ids = [456, 12]

    data = Data(mcc)
    data.injection_mask = mask
    data.projection_mask = mask
    data.get_experiment_data(experiment_ids)

    assert data.centroids.shape == (2, 3)
    assert data.injections.shape[0] == 2
    assert data.projections.shape[0] == 2
    assert data.injections.shape[1] == data.projections.shape[1]
