from __future__ import division
import os
import mock
import pytest
import numpy as np

from numpy.testing import assert_array_equal, assert_raises

from voxel_model.model_data \
        import ModelData, get_experiment_ids, generate_experiments_from_mcc

from voxel_model.experiment import Experiment
from voxel_model.masks import Mask

# =============================================================================
# Module level functions
# -----------------------------------------------------------------------------
# tests
def test_get_experiment_ids(mcc):
    structure_ids = [12, 315]

    experiment_ids = get_experiment_ids(mcc, structure_ids)

    assert( isinstance(experiment_ids, list) )
    assert( all([ isinstance(eid, int) for eid in experiment_ids ]) )

# -----------------------------------------------------------------------------
# tests
def test_generate_experiments_from_mcc(mcc):
    structure_ids = [12, 315]
    experiment_ids = get_experiment_ids(mcc, structure_ids)

    for exp in generate_experiments_from_mcc(mcc, experiment_ids):
        assert( type(exp) == Experiment )


# =============================================================================
# ModelData class
# -----------------------------------------------------------------------------
# tests
def test_is_valid_experiment():
    a = np.ones(5)

    assert( ModelData._is_valid_experiment(a, a, 1) )
    assert( not ModelData._is_valid_experiment(-a, a, 1) )
    assert( not ModelData._is_valid_experiment(a, -a, 1) )
    assert( not ModelData._is_valid_experiment(a, a, -1) )

# -----------------------------------------------------------------------------
# tests
def test_from_mcc_and_masks(mcc):
    source_mask = Mask(mcc, [3], hemisphere=2)
    target_mask = Mask(mcc, [2,3,4,5,6], hemisphere=3)

    data = ModelData.from_mcc_and_masks(mcc, source_mask, target_mask)

    assert( isinstance(data, tuple) )
    assert( isinstance(data.X, np.ndarray) )
    assert( isinstance(data.y, np.ndarray) )
    assert( isinstance(data.source_voxels, np.ndarray) )


# -----------------------------------------------------------------------------
# tests
def test_new():
    x = np.ones((10,103)) # hstack(centroids, injection)
    y = np.ones((10,200))
    a = np.ones((100,3))

    assert( isinstance(ModelData(x,y,a), tuple) )

    # incompatible inner dimensions
    assert_raises( ValueError, ModelData, x, np.ones((5,200)), a )

    # incompatible coords
    assert_raises( ValueError, ModelData, x, y, np.ones((50,3)) )
