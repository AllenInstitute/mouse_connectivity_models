"""
Module containing VoxelModelCache.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

import numpy as np
from allensdk.core import json_utilities
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from .masks import Mask
from .voxel_model_api import VoxelModelApi

from ..models.voxel import VoxelConnectivityArray


class VoxelModelCache(MouseConnectivityCache):
    """Cache class extending MouseConnectivityCache to cache voxel model data.

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

    NODES_KEY = 'NODES'
    WEIGHTS_KEY = 'WEIGHTS'
    SOURCE_MASK_KEY = 'SOURCE_MASK'
    TARGET_MASK_KEY = 'TARGET_MASK'

    CONNECTION_DENSITY_KEY = 'CONNECTION_DENSITY'
    CONNECTION_STRENGTH_KEY = 'CONNECTION_STRENGTH'
    NORMALIZED_CONNECTION_DENSITY_KEY = 'NORMALIZED_CONNECTION_STRENGTH'
    NORMALIZED_CONNECTION_STRENGTH_KEY = 'NORMALIZED_CONNECTION_STRENGTH'

    @classmethod
    def from_json(cls, file_name):
        return cls(**json_utilities.read(file_name))

    def __init__(self,
                 resolution=100,
                 cache=True,
                 manifest_file='voxel_model_manifest.json',
                 ccf_version=None,
                 base_uri=None,
                 version=None):
        super(VoxelModelCache, self).__init__(resolution=resolution,
                                              cache=cache,
                                              manifest_file=manifest_file,
                                              ccf_version=ccf_version,
                                              base_uri=base_uri,
                                              version=version)
        self.api = VoxelModelApi(base_uri=base_uri)

        # MouseConnectivityCache does not set these as attributes so we have
        # to reproduce part of MouseConnectivityCache.__init__() here.
        if version is None:
            version = self.MANIFEST_VERSION
        if ccf_version is None:
            ccf_version = VoxelModelApi.CCF_VERSION_DEFAULT

        self.manifest_file = manifest_file
        self.ccf_version = ccf_version
        self.base_uri = base_uri
        self.version = version

    def get_nodes(self, file_name=None):
        """Get nodes for  from cache."""
        file_name = self.get_cache_path(file_name, self.NODES_KEY)
        self.api.nodes(file_name, strategy='lazy')

        return np.loadtxt(file_name)

    def get_weights(self, file_name=None):
        """Get weights for  from cache."""
        file_name = self.get_cache_path(file_name, self.WEIGHTS_KEY)
        self.api.weights(file_name, strategy='lazy')

        return np.loadtxt(file_name)

    def get_source_mask(self, file_name=None):
        """Get source mask for  from cache."""
        file_name = self.get_cache_path(file_name, self.SOURCE_MASK_KEY)
        self.api.source_mask_params(file_name, strategy='lazy')

        mask_params = json_utilities.read(file_name)

        return Mask.from_cache(self, **mask_params)

    def get_target_mask(self, file_name=None):
        """Get target mask for  from cache."""
        file_name = self.get_cache_path(file_name, self.TARGET_MASK_KEY)
        self.api.target_mask_params(file_name, strategy='lazy')

        mask_params = json_utilities.read(file_name)

        return Mask.from_cache(self, **mask_params)

    def get_voxel_connectivty_array(self,
                                    weights_file_name=None,
                                    nodes_file_name=None,
                                    source_mask_file_name=None,
                                    target_mask_file_name=None):
        """Get  from cache, returning VoxelConnectivityArray.

        Parameters
        ----------
        directory_name : string, optional (default=None)
            Directory to which one wishes to cache the  data. If
            None, the directory_name is pulled from the manifest.
        """
        nodes = self.get_nodes(nodes_file_name)
        weights = self.get_weights(weights_file_name)
        source_mask = self.get_source_mask(source_mask_file_name)
        target_mask = self.get_target_mask(target_mask_file_name)

        return VoxelConnectivityArray(weights, nodes), source_mask, target_mask

    def get_connection_density(self, file_name=None):
        file_name = self.get_cache_path(file_name, self.CONNECTION_DENSITY_KEY)
        self.api.connection_density(file_name, strategy='lazy')

        return np.loadtxt(file_name)

    def get_connection_strength(self, file_name=None):
        file_name = self.get_cache_path(file_name, self.CONNECTION_STRENGTH_KEY)
        self.api.connection_strength(file_name, strategy='lazy')

        return np.loadtxt(file_name)

    def get_normalized_connection_density(self, file_name=None):
        file_name = self.get_cache_path(file_name,
                                        self.NORMALIZED_CONNECTION_DENSITY_KEY)
        self.api.normalized_connection_density(file_name, strategy='lazy')

        return np.loadtxt(file_name)

    def get_normalized_connection_strength(self, file_name=None):
        file_name = self.get_cache_path(file_name,
                                        self.NORMALIZED_CONNECTION_STRENGTH_KEY)
        self.api.normalized_connection_strength(file_name, strategy='lazy')

        return np.loadtxt(file_name)

    def to_json(self, file_name=None):
        params = dict(resolution=self.resolution,
                      cache=self.cache,
                      manifest_file=self.manifest_file,
                      ccf_version=self.ccf_version,
                      base_uri=self.base_uri,
                      version=self.version)

        if file_name is None:
            return json_utilities.write_string(params)

        json_utilities.write(params, file_name)

    def add_manifest_paths(self, manifest_builder):
        """
        Construct a manifest for this Cache class and save it in a file.

        Parameters
        ----------
        file_name: string
            File location to save the manifest.
        """
        manifest_builder = super(VoxelModelCache, self).add_manifest_paths(
            manifest_builder)

        manifest_builder.add_path(self.NODES_KEY,
                                  'voxel_model/nodes.csv.gz',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.WEIGHTS_KEY,
                                  'voxel_model/weights.csv.gz',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.SOURCE_MASK_FILE,
                                  'voxel_model/source_mask.json',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.TARGET_MASK_FILE,
                                  'voxel_model/target_mask.json',
                                  parent_key='BASEDIR',
                                  typename='file')
