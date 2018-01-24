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

from voxel_model.masks import Mask

@pytest.fixture(scope="module")
def structure_ids():
    return [2,3]

@pytest.fixture(scope="module")
def tree():
    return [{'id': 1, 'structure_id_path': [1]},
            {'id': 2, 'structure_id_path': [1, 2]},
            {'id': 3, 'structure_id_path': [1, 3]},
            {'id': 4, 'structure_id_path': [1, 2, 4]},
            {'id': 5, 'structure_id_path': [1, 2, 5]},
            {'id': 6, 'structure_id_path': [1, 2, 5, 6]},
            {'id': 7, 'structure_id_path': [1, 7]}]


@pytest.fixture(scope="module")
def annotation():
    # leaves are 6, 4, 3
    # additionally annotate 2, 5 for realism :)
    annotation = np.zeros((10, 10, 10))
    annotation[4:8, 4:8, 4:8] = 2
    annotation[5:7, 5:7, 5:7] = 5
    annotation[5:7, 5:7, 5] = 6
    annotation[7, 7, 7] = 4
    annotation[8:10, 8:10, 8:10] = 3

    return annotation

@pytest.fixture(scope="module")
def mcc(tree, annotation):
    """Mocks mcc class"""

    mcc = mock.Mock()

    rsp = ReferenceSpace(StructureTree(tree), annotation, [10, 10, 10])
    mcc.get_reference_space.return_value = rsp

    return mcc

@pytest.fixture(scope="function")
def contra_mask(mcc, structure_ids):
    return Mask(mcc, structure_ids, hemisphere=1)

@pytest.fixture(scope="function")
def ipsi_mask(mcc, structure_ids):
    return Mask(mcc, structure_ids, hemisphere=2)

@pytest.fixture(scope="function")
def bi_mask(mcc, structure_ids):
    return Mask(mcc, structure_ids, hemisphere=3)

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
def test_annotation_shape(ipsi_mask, contra_mask, annotation):

    assert( contra_mask.annotation_shape == annotation.shape )
    assert( ipsi_mask.annotation_shape == annotation.shape )

# -----------------------------------------------------------------------------
# tests
def test_coordinates(bi_mask, annotation):

    coordinates = bi_mask.coordinates

    assert( coordinates.shape[1] == len(annotation.shape) )
    assert_array_equal( coordinates, np.argwhere(annotation) )

# -----------------------------------------------------------------------------
# tests
def test_get_key(bi_mask, structure_ids):

    # use the internal structure ids
    base_key = bi_mask.get_key()
    copy_key = bi_mask.get_key(structure_ids)

    assert_array_equal( base_key, copy_key )

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
    volume = np.zeros(bi_mask.annotation_shape)
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
def test_save():
    pass

# -----------------------------------------------------------------------------
# tests
def test_load():
    pass
