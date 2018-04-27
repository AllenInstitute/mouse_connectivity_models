"""
Module containing MCModelsCache.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

import requests
import numpy as np

from allensdk.api.cache import cacheable, Cache
from allensdk.api.mouse_connectivity_api import MouseConnectivityApi
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from .masks import Mask
from ..models.voxel import VoxelConnectivityArray


class MCModelsApi(MouseConnectivityApi):
    '''HTTP Client extending MouseConnectivityApi to download model data.
    '''
    HTTP_MODEL_DIRECTORY = "http://download.alleninstitute.org/publications/"\
            "A_high_resolution_data-driven_model_of_the_mouse_connectome/"

    VOXEL_MODEL_WEIGHTS_FILE = "C57BL-6J_weights.npy"
    VOXEL_MODEL_NODES_FILE = "C57BL-6J_nodes.npy"
    VOXEL_MODEL_SOURCE_MASK_FILE = "source_mask.json"
    VOXEL_MODEL_TARGET_MASK_FILE = "target_mask.json"

    def download_voxel_model_data(self, file_name, save_file_path=None):
        """Download voxel_model data.

        Parameters
        ----------
        file_name : string, optional
        save_file_path : string, optional
            File name to save as.
        """
        url = self.HTTP_MODEL_DIRECTORY + file_name
        self.retrieve_file_over_http(url, save_file_path)

    @cacheable(strategy='create',
               pathfinder=Cache.pathfinder(file_name_position=1,
                                           path_keyword='path'))
    def download_voxel_model_weights(self, file_name):
        pass

    @cacheable(strategy='create',
               pathfinder=Cache.pathfinder(file_name_position=1,
                                           path_keyword='path'))
    def download_voxel_model_nodes(self, file_name):
        pass

    @cacheable(strategy='create',
               pathfinder=Cache.pathfinder(file_name_position=1,
                                           path_keyword='path'))
    def download_voxel_model_source_mask(self, file_name):
        pass

    @cacheable(strategy='create',
               pathfinder=Cache.pathfinder(file_name_position=1,
                                           path_keyword='path'))
    def download_voxel_model_target_mask(self, file_name):
        pass

class MCModelsCache(MouseConnectivityCache):
    """Cache class extending MouseConnectivityCache to cache model data.

    Parameters
    ----------
    resolution: int
        Resolution of grid data to be downloaded when accessing projection volume,
        the annotation volume, and the annotation volume.  Must be one of (10, 25,
        50, 100).  Default is 25.

    ccf_version: string
        Desired version of the Common Coordinate Framework.  This affects the annotation
        volume (get_annotation_volume) and structure masks (get_structure_mask).
        Must be one of (MouseConnectivityApi.CCF_2015, MouseConnectivityApi.CCF_2016).
        Default: MouseConnectivityApi.CCF_2016

    cache: boolean
        Whether the class should save results of API queries to locations specified
        in the manifest file.  Queries for files (as opposed to metadata) must have a
        file location.  If caching is disabled, those locations must be specified
        in the function call (e.g. get_projection_density(file_name='file.nrrd')).

    manifest_file: string
        File name of the manifest to be read.  Default is "mouse_connectivity_manifest.json".
    """

    VOXEL_MODEL_NODES_KEY = 'VOXEL_MODEL_NODES'
    VOXEL_MODEL_WEIGHTS_KEY = 'VOXEL_MODEL_WEIGHTS'

    def __init__(self,
                 resolution=100,
                 cache=True,
                 manifest_file='connectivity/mcmodels_manifest.json',
                 ccf_version=None,
                 base_uri=None,
                 version=None):
        super(MCModelsCache, self).__init__(resolution=resolution,
                                            cache=cache,
                                            manifest_file=manifest_file,
                                            ccf_version=ccf_version,
                                            base_uri=base_uri,
                                            version=version)
        self.api = MCModelsApi(base_uri=base_uri)

    def _get_voxel_model_nodes(self, file_name):
        file_name = self.get_cache_path(file_name, self.VOXEL_MODEL_NODES_KEY)
        self.api.download_model_data(file_name, strategy='lazy')

        return np.loadtxt(file_name)

    def _get_voxel_model_weights(self, file_name):
        file_name = self.get_cache_path(file_name, self.VOXEL_MODEL_WEIGHTS_KEY)
        self.api.download_model_data(file_name, strategy='lazy')

        return np.loadtxt(file_name)

    def _get_voxel_model_source_mask(self, file_name):
        file_name = self.get_cache_path(file_name, self.VOXEL_MODEL_SOURCE_MASK_KEY)
        self.api.download_voxel_model_data(file_name, strategy='lazy')

        return Mask.from_json(file_name)

    def _get_voxel_model_target_mask(self, file_name):
        file_name = self.get_cache_path(file_name, self.VOXEL_MODEL_TARGET_MASK_KEY)
        self.api.download_model_data(file_name, strategy='lazy')

        return Mask.from_json(file_name)

    def get_voxel_model(self, file_name):
        weights = self._get_voxel_model_weights(file_name)
        nodes = self._get_voxel_model_nodes(file_name)
        source_mask = self._get_voxel_model_source_mask(file_name)
        target_mask = self._get_voxel_model_target_mask(file_name)

        return VoxelConnectivityArray(weights, nodes, source_mask, target_mask)

    def get_regionalized_voxel_model(self):
        pass

    def add_manifest_paths(self, manifest_builder):
        """
        Construct a manifest for this Cache class and save it in a file.

        Parameters
        ----------
        file_name: string
            File location to save the manifest.
        """
        manifest_builder = super(MCModelsCache, self).add_manifest_paths(
            manifest_builder)

        manifest_builder.add_path(self.VOXEL_MODEL_COMPONENTS_KEY,
                                  'voxel_model/%s.csv.gz',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.VOXEL_MODEL_REGIONALIZED_KEY,
                                  'voxel_model/%s.csv.gz',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.VOXEL_MODEL_MASK_KEY,
                                  'voxel_model/%s.json',
                                  parent_key='BASEDIR',
                                  typename='file')
