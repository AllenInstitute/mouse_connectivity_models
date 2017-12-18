from __future__ import division
import os
import mock
import pytest
import numpy as np

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.test_utilities.temp_dir import fn_temp_dir

from voxel_model.masks import SourceMask, union_mask

@pytest.fixture(scope="module")
def mcc(fn_temp_dir):

    manifest_path = os.path.join(fn_temp_dir, "manifest.json")
    return MouseConnectivityCache(manifest_file=manifest_path,
                                  resolution=100,
                                  ccf_version="annotation/ccf_2017")

@pytest.fixture(scope="module")
def ccf_shape(mcc):
    return mcc.get_structure_mask(315)[0].shape

@pytest.fixture(scope="module")
def structure_ids(mcc):
    return [315, 313, 512]

@pytest.fixture(scope="function")
def source_mask(mcc, structure_ids):
    return SourceMask(mcc, structure_ids)

def test_union_mask(mcc, structure_ids, ccf_shape):
    umask = union_mask(mcc, structure_ids)

    assert( umask.shape == ccf_shape )
    assert( umask.dtype == bool )

def test_get_mask(source_mask):
    mask = source_mask.mask
    midline = source_mask.ccf_shape[2]//2

    left_hemi = mask[:,:,midline:]

    print left_hemi.sum()
    assert( left_hemi.sum() == 0 )


def test_ccf_shape(ccf_shape, source_mask):

    assert( source_mask.ccf_shape == ccf_shape )

def test_coordinates(source_mask):
    shape = source_mask.coordinates.shape

    assert( shape[1] == 3 )
    assert( shape[0] == int(source_mask.mask.sum()) )

def test_key(source_mask):
    shape = source_mask.key.shape

    assert( len(shape) == 1 )

def test_map_to_ccf(source_mask):
    pass
