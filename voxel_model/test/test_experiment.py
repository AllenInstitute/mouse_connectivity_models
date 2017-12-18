from __future__ import division
import os
import mock
import pytest
import numpy as np

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.test_utilities.temp_dir import md_temp_dir

from voxel_model.experiment import Experiment

@pytest.fixture(scope="module")
def mcc(md_temp_dir):

    manifest_path = os.path.join(md_temp_dir, "manifest.json")
    return MouseConnectivityCache(manifest_file=manifest_path,
                                  resolution=100,
                                  ccf_version="annotation/ccf_2017")

@pytest.fixture(scope="module")
def experiment_id(mcc):
    return 158374671

@pytest.fixture(scope="function")
def experiment(mcc, experiment_id):
    return Experiment(mcc, experiment_id)

def test_mask_to_valid(experiment):
    pass

def test_injection_density(experiment):
    """Covers injection_fraction and projection_density"""
    pass

def test_normalized_projection_density(experiment):
    pass

def test_centroid(experiment):
    pass
