import os

import mock
import pytest

import pandas as pd
import numpy as np
from numpy.testing import assert_allclose

from allensdk.core import json_utilities
from allensdk.test_utilities.temp_dir import fn_temp_dir

from mcmodels.core import VoxelModelCache


@pytest.fixture(scope='function')
def voxel_model_cache(fn_temp_dir, mcc):
    manifest_path = os.path.join(fn_temp_dir, 'voxel_model_manifest.json')
    cache = VoxelModelCache(manifest_file=manifest_path)

    cache.get_reference_space = mock.Mock()
    cache.get_reference_space.return_value = mcc.get_reference_space()

    return cache


def test_from_json(fn_temp_dir):
    # ------------------------------------------------------------------------
    # tests alternative constructor
    manifest_file = 'manifest.json'
    resolution = 10
    path = os.path.join(fn_temp_dir, 'input.json')

    input_data = dict(manifest_file=manifest_file, resolution=resolution)
    json_utilities.write(path, input_data)

    cache = VoxelModelCache.from_json(path)

    assert cache.manifest_file == manifest_file
    assert cache.resolution == resolution


def test_to_json(fn_temp_dir):
    # ------------------------------------------------------------------------
    # tests JSON serialization
    manifest_file = 'manifest.json'
    resolution = 10
    path = os.path.join(fn_temp_dir, 'output.json')

    cache = VoxelModelCache(manifest_file=manifest_file, resolution=resolution)
    cache.to_json(path)

    input_data = json_utilities.read(path)

    assert input_data['manifest_file'] == manifest_file
    assert input_data['resolution'] == resolution


def test_get_nodes(voxel_model_cache, fn_temp_dir):
    # ------------------------------------------------------------------------
    # tests file is cached and properly saved/loaded
    eye = np.eye(100)
    path = os.path.join(fn_temp_dir, 'voxel_model', 'nodes.csv.gz')

    with mock.patch('mcmodels.core.voxel_model_api.VoxelModelApi.'
                    'retrieve_file_over_http',
                    new=lambda a, b, c: np.savetxt(c, eye, delimiter=',')):
        obtained = voxel_model_cache.get_nodes()

    voxel_model_cache.api.retrieve_file_over_http = mock.MagicMock()
    voxel_model_cache.get_nodes()

    voxel_model_cache.api.retrieve_file_over_http.assert_not_called()
    assert_allclose(obtained, eye)
    assert os.path.exists(path)


def test_get_weights(voxel_model_cache, fn_temp_dir):
    # ------------------------------------------------------------------------
    # tests file is cached and properly saved/loaded
    eye = np.eye(100)
    path = os.path.join(fn_temp_dir, 'voxel_model', 'weights.csv.gz')

    with mock.patch('mcmodels.core.voxel_model_api.VoxelModelApi.'
                    'retrieve_file_over_http',
                    new=lambda a, b, c: np.savetxt(c, eye, delimiter=',')):
        obtained = voxel_model_cache.get_weights()

    voxel_model_cache.api.retrieve_file_over_http = mock.MagicMock()
    voxel_model_cache.get_weights()

    voxel_model_cache.api.retrieve_file_over_http.assert_not_called()
    assert_allclose(obtained, eye)
    assert os.path.exists(path)


def test_get_source_mask(voxel_model_cache, fn_temp_dir):
    # ------------------------------------------------------------------------
    # tests file is cached and properly saved/loaded
    structure_ids = [315, 313, 549]
    hemisphere_id = 2
    params = dict(structure_ids=structure_ids, hemisphere_id=hemisphere_id)
    path = os.path.join(fn_temp_dir, 'voxel_model', 'source_mask_params.json')

    with mock.patch('mcmodels.core.voxel_model_api.VoxelModelApi.'
                    'retrieve_file_over_http',
                    new=lambda a, b, c: json_utilities.write(c, params)):
        obtained = voxel_model_cache.get_source_mask()

    voxel_model_cache.api.retrieve_file_over_http = mock.MagicMock()
    voxel_model_cache.get_source_mask()

    voxel_model_cache.api.retrieve_file_over_http.assert_not_called()
    assert obtained.structure_ids == structure_ids
    assert obtained.hemisphere_id == hemisphere_id
    assert os.path.exists(path)


def test_get_target_mask(voxel_model_cache, fn_temp_dir):
    # ------------------------------------------------------------------------
    # tests file is cached and properly saved/loaded
    structure_ids = [315, 313, 549]
    hemisphere_id = 2
    params = dict(structure_ids=structure_ids, hemisphere_id=hemisphere_id)
    path = os.path.join(fn_temp_dir, 'voxel_model', 'target_mask_params.json')

    with mock.patch('mcmodels.core.voxel_model_api.VoxelModelApi.'
                    'retrieve_file_over_http',
                    new=lambda a, b, c: json_utilities.write(c, params)):
        obtained = voxel_model_cache.get_target_mask()

    voxel_model_cache.api.retrieve_file_over_http = mock.MagicMock()
    voxel_model_cache.get_target_mask()

    voxel_model_cache.api.retrieve_file_over_http.assert_not_called()
    assert obtained.structure_ids == structure_ids
    assert obtained.hemisphere_id == hemisphere_id
    assert os.path.exists(path)


def test_get_connection_density(voxel_model_cache, fn_temp_dir):
    # ------------------------------------------------------------------------
    # tests file is cached and properly saved/loaded
    eye = np.eye(100)
    df = pd.DataFrame(eye, index=np.arange(100),
                      columns=pd.MultiIndex.from_product((np.arange(2), np.arange(50))))
    path = os.path.join(fn_temp_dir, 'voxel_model', 'connection_density.csv.gz')

    with mock.patch('mcmodels.core.voxel_model_api.VoxelModelApi.'
                    'retrieve_file_over_http',
                    new=lambda a, b, c: df.to_csv(c, compression='gzip')):
        obtained_ar = voxel_model_cache.get_connection_density(dataframe=False)
        obtained_df = voxel_model_cache.get_connection_density()

    voxel_model_cache.api.retrieve_file_over_http = mock.MagicMock()
    voxel_model_cache.get_connection_density()

    voxel_model_cache.api.retrieve_file_over_http.assert_not_called()
    assert_allclose(obtained_df.values, df.values)
    assert_allclose(obtained_ar, eye)
    assert os.path.exists(path)


def test_get_connection_strength(voxel_model_cache, fn_temp_dir):
    # ------------------------------------------------------------------------
    # tests file is cached and properly saved/loaded
    eye = np.eye(100)
    df = pd.DataFrame(eye, index=np.arange(100),
                      columns=pd.MultiIndex.from_product((np.arange(2), np.arange(50))))
    path = os.path.join(fn_temp_dir, 'voxel_model', 'connection_strength.csv.gz')

    with mock.patch('mcmodels.core.voxel_model_api.VoxelModelApi.'
                    'retrieve_file_over_http',
                    new=lambda a, b, c: df.to_csv(c, compression='gzip')):
        obtained_ar = voxel_model_cache.get_connection_strength(dataframe=False)
        obtained_df = voxel_model_cache.get_connection_strength()

    voxel_model_cache.api.retrieve_file_over_http = mock.MagicMock()
    voxel_model_cache.get_connection_strength()

    voxel_model_cache.api.retrieve_file_over_http.assert_not_called()
    assert_allclose(obtained_df.values, df.values)
    assert_allclose(obtained_ar, eye)
    assert os.path.exists(path)


def test_get_normalized_connection_density(voxel_model_cache, fn_temp_dir):
    # ------------------------------------------------------------------------
    # tests file is cached and properly saved/loaded
    eye = np.eye(100)
    df = pd.DataFrame(eye, index=np.arange(100),
                      columns=pd.MultiIndex.from_product((np.arange(2), np.arange(50))))
    path = os.path.join(fn_temp_dir, 'voxel_model',
                        'normalized_connection_density.csv.gz')

    with mock.patch('mcmodels.core.voxel_model_api.VoxelModelApi.'
                    'retrieve_file_over_http',
                    new=lambda a, b, c: df.to_csv(c, compression='gzip')):
        obtained_ar = voxel_model_cache.get_normalized_connection_strength(dataframe=False)
        obtained_df = voxel_model_cache.get_normalized_connection_density()

    voxel_model_cache.api.retrieve_file_over_http = mock.MagicMock()
    voxel_model_cache.get_normalized_connection_density()

    voxel_model_cache.api.retrieve_file_over_http.assert_not_called()
    assert_allclose(obtained_df.values, df.values)
    assert_allclose(obtained_ar, eye)
    assert os.path.exists(path)


def test_get_normalized_connection_strength(voxel_model_cache, fn_temp_dir):
    # ------------------------------------------------------------------------
    # tests file is cached and properly saved/loaded
    eye = np.eye(100)
    df = pd.DataFrame(eye, index=np.arange(100),
                      columns=pd.MultiIndex.from_product((np.arange(2), np.arange(50))))
    path = os.path.join(fn_temp_dir, 'voxel_model',
                        'normalized_connection_strength.csv.gz')

    with mock.patch('mcmodels.core.voxel_model_api.VoxelModelApi.'
                    'retrieve_file_over_http',
                    new=lambda a, b, c: df.to_csv(c, compression='gzip')):
        obtained_ar = voxel_model_cache.get_normalized_connection_strength(dataframe=False)
        obtained_df = voxel_model_cache.get_normalized_connection_strength()

    voxel_model_cache.api.retrieve_file_over_http = mock.MagicMock()
    voxel_model_cache.get_normalized_connection_strength()

    voxel_model_cache.api.retrieve_file_over_http.assert_not_called()
    assert_allclose(obtained_df.values, df.values)
    assert_allclose(obtained_ar, eye)
    assert os.path.exists(path)
