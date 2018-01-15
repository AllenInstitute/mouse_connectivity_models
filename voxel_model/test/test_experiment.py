from __future__ import division
import os
import mock
import pytest
import numpy as np

from numpy.testing import assert_almost_equal
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.test_utilities.temp_dir import md_temp_dir

from voxel_model.experiment import Experiment, get_model_data
from voxel_model.masks import SourceMask, TargetMask
from voxel_model.utils import get_experiment_ids

@pytest.fixture(scope="module")
def mcc(md_temp_dir):

    manifest_path = os.path.join(md_temp_dir, "manifest.json")
    return MouseConnectivityCache(manifest_file=manifest_path,
                                  resolution=100,
                                  ccf_version="annotation/ccf_2017")

@pytest.fixture(scope="module")
def experiment_id():
    return 100141273

@pytest.fixture(scope="module")
def experiment(mcc, experiment_id):
    return Experiment(mcc, experiment_id)

@pytest.fixture(scope="module")
def normalized_experiment(mcc, experiment_id):
    return Experiment(mcc, experiment_id, normalize_projection=True)

@pytest.fixture(scope="module")
def structure_ids():
    return [512]

@pytest.fixture(scope="module")
def hemisphere():
    return 3

@pytest.fixture(scope="module")
def model_data(mcc, structure_ids, hemisphere):
    # get masks
    source_mask = SourceMask(mcc, structure_ids=structure_ids)
    target_mask = TargetMask(mcc, structure_ids=structure_ids,
                             hemisphere=hemisphere)

    # get experiment ids
    experiment_ids = get_experiment_ids(mcc, structure_ids)

    # return model_data
    return get_model_data(mcc, experiment_ids, source_mask, target_mask)

# ----------------------------------------------------------------------------
# test
def test_Experiment_projection_density(experiment, normalized_experiment):
    # is it actually normalized
    prd = experiment.projection_density
    inj = experiment.injection_density
    normalized_prd = normalized_experiment.projection_density

    assert_almost_equal( prd, normalized_prd*inj.sum() )

# ----------------------------------------------------------------------------
# test
def test_Experiment_centroid(experiment):
    # test centroid inside volume
    centroid = tuple(map(int, np.round(experiment.centroid)))
    centroid_inf = experiment.injection_fraction[centroid]

    # test centroid inside injection
    assert( centroid_inf > 0 )

# ============================================================================
# ----------------------------------------------------------------------------
# test
def test_get_model_data_shapes(model_data):
    # data
    centroids, injections, projections = model_data

    # check shape
    assert( centroids.shape[0] == injections.shape[0] )
    assert( injections.shape[0] == projections.shape[0] )
    assert( projections.shape[0] == centroids.shape[0] )

# ----------------------------------------------------------------------------
# test
def test_get_model_data_OTHER(model_data):
    # NEED MORE TESTS
    pass
