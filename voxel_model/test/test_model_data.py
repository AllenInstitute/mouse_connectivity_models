from __future__ import division
import os
import mock
import pytest
import numpy as np

from numpy.testing import assert_array_equal, assert_raises

from voxel_model.model_data import ModelData, get_experiment_ids

# mock ????
from voxel_model.experiment import Experiment
from voxel_model.masks import Mask

@pytest.fixture(scope="module")
def source_mask(mcc):
    # NOTE : see conftest (annotation and tree
    structure_ids = [3]
    return Mask(mcc, structure_ids, hemisphere=2)

@pytest.fixture(scope="module")
def target_mask(mcc):
    # NOTE : see conftest (annotation and tree
    structure_ids = [2,3,4,5,6]
    return Mask(mcc, structure_ids, hemisphere=3)

# =============================================================================
# Module level functions
# -----------------------------------------------------------------------------
# tests
def test_get_experiment_ids(mcc):
    structure_ids = [12, 315]

    experiment_ids = get_experiment_ids(mcc, structure_ids)

    assert( isinstance(experiment_ids, list) )
    assert( all([ isinstance(eid, int) for eid in experiment_ids ]) )

# =============================================================================
# ModelData class
# -----------------------------------------------------------------------------
# tests
def test_valid_experiment(mcc, source_mask, target_mask):
    pass

# -----------------------------------------------------------------------------
# tests
def test_get_experiment_attrs():
    structure_ids = [12, 315]
    pass

# -----------------------------------------------------------------------------
# tests
def test_from_mcc_and_masks():
    pass

# -----------------------------------------------------------------------------
# tests
def test_new():
    pass
