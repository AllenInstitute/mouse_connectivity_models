import os

import mock
import pytest

import numpy as np
from numpy.testing import assert_allclose

from allensdk.test_utilities.temp_dir import temp_dir

from mcmodels.core import VoxelModelApi

@pytest.fixture(scope="function")
def fn_temp_dir(request):
    return temp_dir(request)

@pytest.fixture(scope='function')
def voxel_model_api():
    return VoxelModelApi()


def test_download_nodes(voxel_model_api):
    # ------------------------------------------------------------------------
    # test proper key passed
    with mock.patch('mcmodels.core.voxel_model_api.VoxelModelApi.'
            'download_model_files') as api_dmf:
        voxel_model_api.download_nodes('file_name')

        api_dmf.assert_called_once_with("nodes.csv.gz", 'file_name')


def test_download_weights(voxel_model_api):
    # ------------------------------------------------------------------------
    # test proper key passed
    with mock.patch('mcmodels.core.voxel_model_api.VoxelModelApi.'
            'download_model_files') as api_dmf:
        voxel_model_api.download_weights('file_name')

        api_dmf.assert_called_once_with("weights.csv.gz", 'file_name')


def test_download_source_mask_params(voxel_model_api):
    # ------------------------------------------------------------------------
    # test proper key passed
    with mock.patch('mcmodels.core.voxel_model_api.VoxelModelApi.'
            'download_model_files') as api_dmf:
        voxel_model_api.download_source_mask_params('file_name')

        api_dmf.assert_called_once_with("source_mask_params.json", 'file_name')


def test_download_target_mask_params(voxel_model_api):
    # ------------------------------------------------------------------------
    # test proper key passed
    with mock.patch('mcmodels.core.voxel_model_api.VoxelModelApi.'
            'download_model_files') as api_dmf:
        voxel_model_api.download_target_mask_params('file_name')

        api_dmf.assert_called_once_with("target_mask_params.json", 'file_name')


def test_download_connection_density(voxel_model_api):
    # ------------------------------------------------------------------------
    # test proper key passed
    with mock.patch('mcmodels.core.voxel_model_api.VoxelModelApi.'
            'download_model_files') as api_dmf:
        voxel_model_api.download_connection_density('file_name')

        api_dmf.assert_called_once_with('connection_density.csv.gz', 'file_name')

def test_download_connection_strength(voxel_model_api):
    # ------------------------------------------------------------------------
    # test proper key passed
    with mock.patch('mcmodels.core.voxel_model_api.VoxelModelApi.'
            'download_model_files') as api_dmf:
        voxel_model_api.download_connection_strength('file_name')

        api_dmf.assert_called_once_with('connection_strength.csv.gz', 'file_name')


def test_download_normalized_connection_density(voxel_model_api):
    # ------------------------------------------------------------------------
    # test proper key passed
    with mock.patch('mcmodels.core.voxel_model_api.VoxelModelApi.'
            'download_model_files') as api_dmf:
        voxel_model_api.download_normalized_connection_density('file_name')

        api_dmf.assert_called_once_with('normalized_connection_density.csv.gz',
                                        'file_name')


def test_download_normalized_connection_strength(voxel_model_api):
    # ------------------------------------------------------------------------
    # test proper key passed
    with mock.patch('mcmodels.core.voxel_model_api.VoxelModelApi.'
            'download_model_files') as api_dmf:
        voxel_model_api.download_normalized_connection_strength('file_name')

        api_dmf.assert_called_once_with('normalized_connection_strength.csv.gz',
                                        'file_name')
