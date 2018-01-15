from __future__ import division
import os
import mock
import pytest
import numpy as np

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
    return 158374671

@pytest.fixture(scope="module")
def structure_ids():
    return [512]

@pytest.fixture(scope="module")
def hemisphere():
    return 3

# ----------------------------------------------------------------------------
# test
def test_mask_to_valid():
    pass

# ----------------------------------------------------------------------------
# test
def test_get_model_data(mcc, structure_ids, hemisphere):
    # get masks
    source_mask = SourceMask(mcc, structure_ids=structure_ids)
    target_mask = TargetMask(mcc, structure_ids=structure_ids,
                             hemisphere=hemisphere)

    # get experiment ids
    experiment_ids = get_experiment_ids(mcc, structure_ids)

    centroids, injections, projections = get_model_data(mcc,
                                                        experiment_ids,
                                                        source_mask,
                                                        target_mask)

    # check shape
    assert( centroids.shape[0] == injections.shape[0] )
    assert( injections.shape[0] == projections.shape[0] )
    assert( projections.shape[0] == centroids.shape[0] )
