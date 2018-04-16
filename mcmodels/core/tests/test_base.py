from __future__ import division

import mock
import pytest
import numpy as np

from numpy.testing import assert_array_equal, assert_raises

from mcmodels.core import VoxelData
from mcmodels.core import RegionalData
from mcmodels.core.tests.conftest import mcc


@pytest.fixture(scope="function")
def voxel_data():
    data = VoxelData
    data.default_structure_ids = [5, 6]
    return data


# =============================================================================
# VoxelData class
# =============================================================================
def test_default_structure_ids():
    # TODO not yet tested in sdk, assume solid for now
    pass


def test_experiment_generator(mcc, voxel_data):
    def get_experiment_size(data):
        size = 0
        for _ in data._experiment_generator(experiment_ids):
            size += 1
        return size
    # ------------------------------------------------------------------------
    # tests pulls all with no option
    experiment_ids = [456, 12]

    # full
    data = voxel_data(mcc)
    exps = get_experiment_size(data)

    assert exps == 2

    # ------------------------------------------------------------------------
    # tests pulls none when min_inj_volume too high
    data = voxel_data(mcc, injection_volume_bounds=(np.inf, 0))
    exps = get_experiment_size(data)

    assert exps == 0

    # ------------------------------------------------------------------------
    # tests pulls none when hemi id is invalid
    data = voxel_data(mcc, injection_hemisphere_id=-5, flip_experiments=False)
    exps = get_experiment_size(data)

    assert exps == 0

    # TODO: flip option
    # data = _BaseModelData(mcc, injection_hemisphere_id=1, flip_experiments=True)


def test_get_experiment_data(voxel_data, mcc, mask):
    # ------------------------------------------------------------------------
    # tests data arrays are set as attrs and have proper shape
    experiment_ids = [456, 12]

    data = voxel_data(mcc)
    data.get_experiment_data(experiment_ids)

    assert data.centroids.shape == (2, 3)
    assert data.injections.shape[0] == 2
    assert data.projections.shape[0] == 2
    assert data.injections.shape[1] == data.projections.shape[1]


def test_get_regional_model(voxel_data, mcc, mask):
    # ------------------------------------------------------------------------
    # tests returns regional data
    experiment_ids = [456, 12]

    data = voxel_data(mcc)
    regional_data = data.get_regional_data()

    assert isinstance(regional_data, RegionalData)
    assert not hasattr(regional_data, 'centroids')

    # ------------------------------------------------------------------------
    # tests sets centroids, injections, projections
    data.get_experiment_data(experiment_ids)
    regional_data = data.get_regional_data()

    assert hasattr(regional_data, 'centroids')
    assert hasattr(regional_data, 'injections')
    assert hasattr(regional_data, 'projections')


# =============================================================================
# RegionalData class
# =============================================================================
def test_unionize_experiment_data(voxel_data, mcc):
    # ------------------------------------------------------------------------
    # tests injections, projections regionalized
    experiment_ids = [456, 12]

    data = voxel_data(mcc)
    data.get_experiment_data(experiment_ids)
    regional_data = data.get_regional_data()

    # 2 regions in mask
    assert regional_data.injections.shape[1] == 2
    assert regional_data.projections.shape[1] == 2
