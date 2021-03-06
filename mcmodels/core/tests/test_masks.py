"""
tree, annotataion taken directly (copied) from:
    allensdk.test.core.test_reference_space.rsp()
"""
# TODO : test load/save
from __future__ import division

import mock
import pytest
import numpy as np

from numpy.testing import assert_array_equal, assert_raises
from allensdk.core.reference_space import ReferenceSpace
from allensdk.core.structure_tree import StructureTree

from mcmodels.core import Mask
from mcmodels.core.tests.conftest import mcc, annotation

@pytest.fixture(scope="module")
def structure_ids():
    return [2,3]

@pytest.fixture(scope="module")
def descendant_ids():
    return [2,3,4,5,6]

@pytest.fixture(scope="function")
def contra_mask(mcc, structure_ids):
    return Mask.from_cache(mcc, structure_ids=structure_ids, hemisphere_id=1)

@pytest.fixture(scope="function")
def ipsi_mask(mcc, structure_ids):
    return Mask.from_cache(mcc, structure_ids=structure_ids, hemisphere_id=2)

@pytest.fixture(scope="function")
def bi_mask(mcc, structure_ids):
    return Mask.from_cache(mcc, structure_ids=structure_ids, hemisphere_id=3)


def test_repr(mcc):
    # ------------------------------------------------------------------------
    # test structure_ids <= 3
    structure_ids = list(range(3))
    mask = Mask.from_cache(mcc, structure_ids=structure_ids, hemisphere_id=3)

    s = "Mask(hemisphere_id=3, structure_ids=[0, 1, 2])"
    assert repr(mask) == s

    # ------------------------------------------------------------------------
    # test structure_ids > 3
    structure_ids = list(range(10))
    mask = Mask.from_cache(mcc, structure_ids=structure_ids, hemisphere_id=3)

    s = "Mask(hemisphere_id=3, structure_ids=[0, ..., 9])"
    assert repr(mask) == s

# -----------------------------------------------------------------------------
# tests
def test_mask_to_hemisphere(ipsi_mask, contra_mask, bi_mask):

    midline = bi_mask.mask.shape[2]//2

    ipsi_left_hemi = ipsi_mask.mask[:,:,:midline]
    contra_right_hemi = contra_mask.mask[:,:,midline:]

    assert( ipsi_left_hemi.sum() == 0 )
    assert( contra_right_hemi.sum() == 0 )

    combo = ipsi_mask.mask + contra_mask.mask


    assert_array_equal( combo, bi_mask.mask )

# -----------------------------------------------------------------------------
# tests
def test_get_mask(bi_mask, annotation):


    assert_array_equal( bi_mask.mask, annotation != 0 )

# -----------------------------------------------------------------------------
# tests
def test_assigned_structures(bi_mask, ipsi_mask, annotation, descendant_ids):

    assert( bi_mask.assigned_structures == set(descendant_ids) )

    ipsi_annot = annotation[...,annotation.shape[2]//2:]
    ipsi_defined = np.unique(ipsi_annot[ipsi_annot.nonzero()]).astype(int)
    assert( ipsi_mask.assigned_structures == set(ipsi_defined) )


# -----------------------------------------------------------------------------
# tests
def test_coordinates(bi_mask, annotation):

    coordinates = bi_mask.coordinates

    assert( coordinates.shape[1] == len(annotation.shape) )
    assert_array_equal( coordinates, np.argwhere(annotation) )

# -----------------------------------------------------------------------------
# tests
def test_get_flattened_voxel_index(bi_mask, annotation):
    full_key = annotation[annotation.nonzero()]

    v1 = [7,7,7] # = 4
    v2 = [9,9,9] # = 3
    i1 = np.where(full_key == 4)[0][0]
    i2 = np.where(full_key == 3)[0][7]

    print( bi_mask.get_flattened_voxel_index(v1), i1 )
    print( bi_mask.get_flattened_voxel_index(v2), i2 )
    assert( bi_mask.get_flattened_voxel_index(v1) == i1 )
    assert( bi_mask.get_flattened_voxel_index(v2) == i2 )

    assert_raises( ValueError, bi_mask.get_flattened_voxel_index, [10,10,10] )
    assert_raises( ValueError, bi_mask.get_flattened_voxel_index, [5, 5.5, 5] )

# -----------------------------------------------------------------------------
# tests
def test_get_key(bi_mask, structure_ids):

    # use the internal structure ids
    base_key = bi_mask.get_key()
    copy_key = bi_mask.get_key(structure_ids)

    #assert_array_equal( base_key, copy_key )

    # check root covers
    root_key = bi_mask.get_key( [1] )

    assert_array_equal( base_key != 0, root_key != 0 )

    # leaves (mutually exclusive coverages)
    all_leaves_key = bi_mask.get_key( [3,4,6] )
    single_leaf_key = bi_mask.get_key( [4] )

    assert_array_equal( base_key == 3, all_leaves_key == 3 )
    assert_array_equal( all_leaves_key == 4, single_leaf_key != 0 )

    # test all the same shape and coverage
    keys = [copy_key, root_key, all_leaves_key, single_leaf_key]

    assert( all([key.shape == base_key.shape for key in keys]) )

    # test disjoint error
    assert_raises( ValueError, bi_mask.get_key, [2,3,4,5,6] )

# -----------------------------------------------------------------------------
# tests
def test_mask_volume(bi_mask, annotation):

    y = np.ones((annotation.shape))
    y_masked = bi_mask.mask_volume(y)

    assert( y_masked.shape == (bi_mask.mask.sum(),) )

    masked_annotation = bi_mask.mask_volume(annotation)

    assert_array_equal( masked_annotation, annotation[annotation.nonzero()] )

# -----------------------------------------------------------------------------
# tests
def test_fill_volume_where_masked(bi_mask):

    # copy/val fill
    volume = np.zeros(bi_mask.reference_space.annotation.shape)
    val_filled = bi_mask.fill_volume_where_masked(volume, 9, inplace=False)

    assert_array_equal( np.unique(val_filled), np.array([0,9]) )
    assert( val_filled.sum() == 9*bi_mask.mask.sum())

    # inplace/array fill
    fill_arr = np.arange(bi_mask.coordinates.shape[0])
    bi_mask.fill_volume_where_masked(volume, fill_arr, inplace=True)

    assert_array_equal( np.unique(volume), fill_arr )

    # test mismatch
    args = (volume, range(11))
    assert_raises(ValueError, bi_mask.fill_volume_where_masked, *args )

# -----------------------------------------------------------------------------
# tests
def test_map_masked_to_annotation(bi_mask, annotation):

    masked = bi_mask.mask_volume(annotation)

    assert_array_equal( bi_mask.map_masked_to_annotation(masked), annotation )


# -----------------------------------------------------------------------------
# tests
def get_injection_ratio_contained(experiment):
    # np.ndarray
    mask = np.ones_like(experiment.injection_density)
    mask[..., :mask.shape[2]//2] = 0

    assert( experiment.get_injection_ratio_contained(mask) == 0.5 )

    # Mask object
    mask = Mask(mcc, [2,3], hemisphere_id=3)
    assert( type(experiment.get_injection_ratio_contained(mask)) == float )

    # wrong np.ndarray size
    assert_raises(ValueError, experiment.get_injection_density, np.ones((2,2)))

# -----------------------------------------------------------------------------
# tests
def mask_volume(experiment):
    # np.ndarray
    mask = np.ones_like(experiment.injection_density)
    mask[..., :mask.shape[2]//2] = 0

    n_nnz = len(mask.nonzero()[0])
    assert(experiment.mask_volume("injection_density", mask).shape == (n_nnz,))

    # Mask object
    mask = Mask(mcc, [2,3], hemisphere_id=3)
    assert( len(experiment.mask_volume("injection_density", mask).shape) == 1  )

    # wrong np.ndarray size
    assert_raises( ValueError, experiment.mask_volume, "data", mask )
    assert_raises( ValueError, experiment.mask_volume, "injection_density",
                   np.ones((2, 2)))
