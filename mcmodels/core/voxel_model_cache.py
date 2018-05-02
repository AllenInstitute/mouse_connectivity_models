"""
Module containing VoxelModelCache.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

import numpy as np
import pandas as pd
from allensdk.core import json_utilities
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from .base import VoxelData
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

    Examples
    --------
    >>> from mcmodels.core import VoxelModelCache
    >>> cache = VoxelModelCache(manifest_file='connectivity/voxel_model_manifest.json')
    >>> # download the fitted voxel-scale model
    >>> voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()
    >>> # get regionalized adjacency matrix
    >>> regional_ncd = cache.get_normalized_connection_density()
    >>> # get VoxelData object to use in model building
    >>> cortex_voxel_data = cache.get_experiment_data(injection_structure_ids=[315])
    >>> # JSON serialize input parameters so that cache can be reinstantiated later
    >>> cache.to_json()
    '{
            "resolution": 100,
            "cache": true,
            "manifest_file": "voxel_model_manifest.json",
            "ccf_version": "annotation/ccf_2017",
            "base_uri": null,
            "version": 1.2
    }'
    """

    COARSE_STRUCTURE_SET_ID = 2
    DEFAULT_STRUCTURE_SET_IDS = tuple([COARSE_STRUCTURE_SET_ID])

    NODES_KEY = 'NODES'
    WEIGHTS_KEY = 'WEIGHTS'
    SOURCE_MASK_KEY = 'SOURCE_MASK'
    TARGET_MASK_KEY = 'TARGET_MASK'

    CONNECTION_DENSITY_KEY = 'CONNECTION_DENSITY'
    CONNECTION_STRENGTH_KEY = 'CONNECTION_STRENGTH'
    NORMALIZED_CONNECTION_DENSITY_KEY = 'NORMALIZED_CONNECTION_DENSITY'
    NORMALIZED_CONNECTION_STRENGTH_KEY = 'NORMALIZED_CONNECTION_STRENGTH'

    @property
    def default_structure_ids(self):
        """Default structure ids."""
        # NOTE: Necessary copy from allensdk.core.MouseConnectivityCache because
        #       of hardcoded class and summary structure set id error due to
        #       new annotation (ccf)

        if not hasattr(self, '_default_structure_ids'):
            tree = self.get_structure_tree()
            default_structures = tree.get_structures_by_set_id(
                self.DEFAULT_STRUCTURE_SET_IDS)
            self._default_structure_ids = [st['id'] for st in default_structures
                                           if st['id'] != 934]

        return self._default_structure_ids

    @classmethod
    def from_json(cls, file_name):
        """Construct object from JSON serialized parameter file.

        Parameters
        ----------
        file_name : string
            Path to .json file containing VoxelModelCache parameters.

        Returns
        -------
        A VoxelModelCache object
        """
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
        """Get nodes for voxel-scale model from cache.

        Parameters
        ----------
        file_name: string, optional (default=None)
            File name to store the voxel model nodes.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.

        Returns
        -------
        nodes (type=numpy.ndarray)
        """
        file_name = self.get_cache_path(file_name, self.NODES_KEY)
        self.api.download_nodes(file_name, strategy='lazy')

        return np.loadtxt(file_name, delimiter=',')

    def get_weights(self, file_name=None):
        """Get weights for voxel-scale model from cache.

        Parameters
        ----------
        file_name: string, optional (default=None)
            File name to store the voxel model weights.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.

        Returns
        -------
        weights (type=numpy.ndarray)
        """
        file_name = self.get_cache_path(file_name, self.WEIGHTS_KEY)
        self.api.download_weights(file_name, strategy='lazy')

        return np.loadtxt(file_name, delimiter=',')

    def get_source_mask(self, file_name=None):
        """Get source mask for voxel-scale model from cache.

        Parameters
        ----------
        file_name: string, optional (default=None)
            File name to store the voxel model source_mask.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.

        Returns
        -------
        Mask object.
        """
        file_name = self.get_cache_path(file_name, self.SOURCE_MASK_KEY)
        self.api.download_source_mask_params(file_name, strategy='lazy')

        mask_params = json_utilities.read(file_name)

        return Mask.from_cache(self, **mask_params)

    def get_target_mask(self, file_name=None):
        """Get target mask for voxel-scale model from cache.

        Parameters
        ----------
        file_name: string, optional (default=None)
            File name to store the voxel model target_mask.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.

        Returns
        -------
        Mask object.
        """
        file_name = self.get_cache_path(file_name, self.TARGET_MASK_KEY)
        self.api.download_target_mask_params(file_name, strategy='lazy')

        mask_params = json_utilities.read(file_name)

        return Mask.from_cache(self, **mask_params)

    def get_voxel_connectivity_array(self,
                                     nodes_file_name=None,
                                     weights_file_name=None,
                                     source_mask_file_name=None,
                                     target_mask_file_name=None):
        """Get voxel-scale model from cache, returning VoxelConnectivityArray.

        Parameters
        ----------
        nodes_file_name : string, optional (default=None)
            File name to store the voxel model nodes. See :meth:`get_nodes`

        weights_file_name : string, optional (default=None)
            File name to store the voxel model weights. See :meth:`get_weights`

        source_mask_file_name : string, optional (default=None)
            File name to store the source_mask. See :meth:`get_source_mask`

        target_mask_file_name : string, optional (default=None)
            File name to store the target_mask. See :meth:`get_target_mask`

        Returns
        -------
        tuple : (VoxelConnectivityArray, Mask, Mask)
            (get_voxel_connectivity_array, source_mask, target_mask)
        """
        nodes = self.get_nodes(nodes_file_name)
        weights = self.get_weights(weights_file_name)
        source_mask = self.get_source_mask(source_mask_file_name)
        target_mask = self.get_target_mask(target_mask_file_name)

        return VoxelConnectivityArray(weights, nodes), source_mask, target_mask

    def get_connection_density(self, file_name=None, dataframe=True):
        """Get regionalized voxel-model weights as :term:`connection density`.

        Parameters
        ----------
        file_name: string, optional (default = None)
            File name to store the voxel model target_mask.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.

        dataframe: boolean, optional (default = True)
            Return the regionalized weights as a Pandas DataFrame. If False,
            return an unlabeled `numpy.ndarray`.
        """
        file_name = self.get_cache_path(file_name, self.CONNECTION_DENSITY_KEY)
        self.api.download_connection_density(file_name, strategy='lazy')

        df = pd.read_csv(file_name, header=[0, 1], index_col=[0])

        if dataframe:
            return df
        return df.values

    def get_connection_strength(self, file_name=None, dataframe=True):
        """Get regionalized voxel-model weights as :term:`connection strength`.

        Parameters
        ----------
        file_name: string, optional (default = None)
            File name to store the voxel model target_mask.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.

        dataframe: boolean, optional (default = True)
            Return the regionalized weights as a Pandas DataFrame. If False,
            return an unlabeled `numpy.ndarray`.
        """
        file_name = self.get_cache_path(file_name, self.CONNECTION_STRENGTH_KEY)
        self.api.download_connection_strength(file_name, strategy='lazy')

        df = pd.read_csv(file_name, header=[0, 1], index_col=[0])

        if dataframe:
            return df
        return df.values

    def get_normalized_connection_density(self, file_name=None, dataframe=True):
        """Get regionalized voxel-model weights as :term:`normalized connection density`.

        Parameters
        ----------
        file_name: string, optional (default = None)
            File name to store the voxel model target_mask.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.

        dataframe: boolean, optional (default = True)
            Return the regionalized weights as a Pandas DataFrame. If False,
            return an unlabeled `numpy.ndarray`.
        """
        file_name = self.get_cache_path(file_name,
                                        self.NORMALIZED_CONNECTION_DENSITY_KEY)
        self.api.download_normalized_connection_density(file_name, strategy='lazy')

        df = pd.read_csv(file_name, header=[0, 1], index_col=[0])

        if dataframe:
            return df
        return df.values

    def get_normalized_connection_strength(self, file_name=None, dataframe=True):
        """Get regionalized voxel-model weights as :term:`normalized connection strength`.

        Parameters
        ----------
        file_name: string, optional (default = None)
            File name to store the voxel model target_mask.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.

        dataframe: boolean, optional (default = True)
            Return the regionalized weights as a Pandas DataFrame. If False,
            return an unlabeled `numpy.ndarray`.
        """
        file_name = self.get_cache_path(file_name,
                                        self.NORMALIZED_CONNECTION_STRENGTH_KEY)
        self.api.download_normalized_connection_strength(file_name, strategy='lazy')

        df = pd.read_csv(file_name, header=[0, 1], index_col=[0])

        if dataframe:
            return df
        # else return numpy.ndarray
        return df.values

    def get_experiment_data(self, cre=None, injection_structure_ids=None, **kwargs):
        """Pulls voxel-scale grid data for experiments.

        Parameters
        ----------
        cre: boolean or list
            If True, return only cre-positive experiments.  If False, return only
            cre-negative experiments.  If None, return all experiments. If list, return
            all experiments with cre line names in the supplied list. Default None.

        cache - VoxelModelCache or MouseConnectivityCache object
            Provides way to pull experiment grid-data from Allen Brain Atlas

        injection_structure_ids : list, optional, default None
            List of structure_ids to which the injection mask will be constrained.

        projection_structure_ids : list, optional, default None
            List of structure_ids to which the projection mask will be constrained.

        injection_hemisphere_id : int, optional, default 3
            Hemisphere (1:left, 2:right, 3:both) to which the injection mask will
            be constrained.

        projection_hemisphere_id : int, optional, default 3
            Hemisphere (1:left, 2:right, 3:both) to which the projection mask will
            be constrained.

        normalized_injection : boolean, optional, default True
            If True, the injection density will be normalized by the total
            injection density for each experiment.

        normalized_projection : boolean, optional, default True
            If True, the projection density will be normalized by the total
            injection density for each experiment.

        flip_experiments : boolean, optional, default True
            If True, experiment grid-data will be reflected across the midline.
            Useful if you wish to include L hemisphere injections into a R
            hemisphere model.

        data_mask_tolerance : float, optional, default 0.0
            Tolerance with which to include data in voxels informatically labeled
            as having error. The data_mask for each experiment is an array with
            values between (0, 1), where 1 indicates the voxel fully contains an
            error, whereas 0 indicates the voxel does not contain any error. A value
            of 0.0 thus indicates the highest threshold for data, whereas a value of
            1.0 indicates that data will be included from all voxels.

        injection_volume_bounds : float, optional, default (0.0, np.inf)
            Includes experiments with total injection volume (mm^3) within bounds.

        projection_volume_bounds : float, optional, default (0.0, np.inf)
            Includes experiments with total projection volume (mm^3) within bounds.

        min_contained_injection_ratio : float, optional, default 0.0
            Includes experiments with total injection volume ratio within injection
            mask.

        Returns
        -------
        A VoxelData object with attributes centroids, injections, projections.

        See Also
        --------
        VoxelData.get_experiment_data
        """
        if injection_structure_ids is None:
            injection_structure_ids = self.default_structure_ids

        experiment_ids = [e['id'] for e in self.get_experiments(
            dataframe=False, cre=cre, injection_structure_ids=injection_structure_ids)]

        voxel_data = VoxelData(
            self, injection_structure_ids=injection_structure_ids, **kwargs)

        return voxel_data.get_experiment_data(experiment_ids)

    def to_json(self, file_name=None):
        """JSON serialize object parameters to file or string.

        Parameters
        ----------
        file_name : string, optional (default None)
            Path to .json file containing VoxelModelCache parameters. If None,
            a string will be returned.

        Returns
        -------
        string
            If file_name == None, a string of the JSON serialization is returned.
        """
        params = dict(resolution=self.resolution,
                      cache=self.cache,
                      manifest_file=self.manifest_file,
                      ccf_version=self.ccf_version,
                      base_uri=self.base_uri,
                      version=self.version)

        if file_name is None:
            return json_utilities.write_string(params)

        json_utilities.write(file_name, params)

    def add_manifest_paths(self, manifest_builder):
        """Construct a manifest for this Cache class and save it in a file."""
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

        manifest_builder.add_path(self.SOURCE_MASK_KEY,
                                  'voxel_model/source_mask_params.json',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.TARGET_MASK_KEY,
                                  'voxel_model/target_mask_params.json',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.CONNECTION_DENSITY_KEY,
                                  'voxel_model/connection_density.csv.gz',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.CONNECTION_STRENGTH_KEY,
                                  'voxel_model/connection_strength.csv.gz',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.NORMALIZED_CONNECTION_DENSITY_KEY,
                                  'voxel_model/normalized_connection_density.csv.gz',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.NORMALIZED_CONNECTION_STRENGTH_KEY,
                                  'voxel_model/normalized_connection_strength.csv.gz',
                                  parent_key='BASEDIR',
                                  typename='file')

        return manifest_builder
