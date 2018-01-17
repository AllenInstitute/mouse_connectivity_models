from __future__ import division
import mock
import copy
import pytest
import numpy as np

from itertools import cycle
from numpy.testing import assert_array_equal
from voxel_model.masks import SourceMask, TargetMask, union_mask

@pytest.fixture(scope="module")
def structure_ids():
    # arbitrary
    return [101, 310]

@pytest.fixture(scope="function")
def mcc():
    mask_r = ( np.array([[[1,1],[1,0]],[[1,0],[0,0]]]), )
    mask_l = ( np.array([[[0,0],[0,1]],[[0,1],[1,1]]]), )

    mcc = mock.Mock()
    mcc.get_structure_mask.side_effect = cycle([mask_r, mask_l])
    return mcc

@pytest.fixture(scope="function")
def source_mask(mcc, structure_ids):
    return SourceMask(mcc, structure_ids)

# -----------------------------------------------------------------------------
# tests
def test_union_mask(mcc):
    structure_ids = [101, 310] #arbitrary
    mask = union_mask(mcc, structure_ids, return_key=False)
    key = union_mask(mcc, structure_ids, return_key=True)

    # mask should cover whole
    assert_array_equal( mask, np.ones((2,2,2)) )

    # same coverage
    assert_array_equal( (mask > 0), (key > 0) )

    # keys should only be sids above
    assert_array_equal( np.unique(keys), structure_ids )

# _BaseMask, SourceMask
# -----------------------------------------------------------------------------
# tests
def test_get_mask(source_mask):
    left_hemi = source_mask.mask[:,:,1:]

    assert( left_hemi.sum() == 0 )

# -----------------------------------------------------------------------------
# tests
def test_ccf_shape(source_mask):

    assert( source_mask.ccf_shape == (2,2,2) )

# -----------------------------------------------------------------------------
# tests
def test_coordinates(source_mask):
    shape = source_mask.coordinates.shape

    assert( shape[1] == 3 )
    assert( shape[0] == int(source_mask.mask.sum()) )

# -----------------------------------------------------------------------------
# tests
def test_key(source_mask):
    shape = source_mask.key.shape

    assert( len(shape) == 1 )

# -----------------------------------------------------------------------------
# tests
def test_map_to_ccf(source_mask):
    """NOT SUFFICIENT"""
    mask_ = source_mask.mask.astype(np.int)
    y = np.ones_like(source_mask.key)

    assert_array_equal( source_mask.map_to_ccf(y), source_mask.mask )

# TargetMask
# -----------------------------------------------------------------------------
# tests constructor
# TODO: change to classmethod constructor
def test_target_mask(mcc, structure_ids):
    ipsi = TargetMask(mcc, structure_ids, "ipsi")
    contra = TargetMask(mcc, structure_ids, "contra")
    both = TargetMask(mcc, structure_ids, "both")

    # constructor
    assert( ipsi.hemisphere == 2 )
    assert( contra.hemisphere == 1 )
    assert( both.hemisphere == 3 )

    # hemi masks
    assert( ipsi.mask[:,:,1:].sum() == 0 )
    assert( contra.mask[:,:,:1].sum() == 0 )
