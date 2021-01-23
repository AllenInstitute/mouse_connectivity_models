# """
# Module containing the RegionalizedModel object, used in evaluating the
# voxel-scale model at the level of regions.
# """
# # Authors: Joseph Knox <josephk@alleninstitute.org>
# # License: Allen Institute Software License
#
# from __future__ import division, absolute_import
#
# import numpy as np
# import pandas as pd
#
# from ...utils import nonzero_unique, unionize
#
#
# class RegionalizedModel(object):
#     """Regionalization/Parcelation of VoxelModel.
#
#     Regionalizes the connectivity model in VoxelModel given a brain parcelation.
#
#     Parameters
#     ----------
#     source_key : array-like, shape=(n_source_voxels,)
#         Flattened key relating each source voxel to a given brain region.
#
#     target_key : array-like, shape=(n_target_voxels,)
#         Flattened key relating each target voxel to a given brain region.
#
#     ordering : array-like, optional (default=None)
#         Order with which to arrange the source/target regions. If supplied, the
#         ordering must contain at least every unique structure_id associated with
#         each of the source/target regions.
#
#     dataframe : boolean, optional (default=False)
#         If True, each metric of the regionalized model will be returned as a
#         labeled pandas dataframe. Else, each metric will be returned as an
#         unlabeled numpy array.
#
#     Examples
#     --------
#     >>> from mcmodels.core import VoxelModelCache
#     >>> from mcmodels.models.voxel import RegionalizedModel
#     >>> cache = VoxelModelCache()
#     >>> # pull voxel-scale model from cache
#     >>> voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()
#     >>> # regionalize to summary structures (region 934 was removed in new ccf)
#     >>> regions = cache.get_structures_by_set_id(
#     >>> region_ids = [r['id'] for r in regions if r['id'] != 934]
#     >>> # get array keys
#     >>> source_key = source_mask.get_key(region_ids)
#     >>> target_key = source_mask.get_key(region_ids)
#     >>> # regionalize model
#     >>> regional_model = RegionalizedModel.from_voxel_array(
#     ...     voxel_array, source_key, target_key)
#     >>> regional_model.normalized_connection_density.shape
#     (291, 577)
#     """
#
#     VALID_REGION_METRICS = [
#         "connection_strength",
#         "connection_density",
#         "normalized_connection_strength",
#         "normalized_connection_density"
#     ]
#
#     @classmethod
#     def from_voxel_array(cls, voxel_array, *args, **kwargs):
#         """Alternative constructor.
#
#         Parameters
#         ----------
#         voxel_array : VoxelConnectivityArray object
#             The voxel-scale model in the form of a VoxelConnectivityArray object.
#         source_key : array-like, shape=(n_source_voxels,)
#             Flattened key relating each source voxel to a given brain region.
#
#         target_key : array-like, shape=(n_target_voxels,)
#             Flattened key relating each target voxel to a given brain region.
#
#         ordering : array-like, optional (default=None)
#             Order with which to arrange the source/target regions. If supplied,
#             the ordering must contain at least every unique structure_id
#             associated with each of the source/target regions.
#
#         dataframe : boolean, optional (default=False)
#             If True, each metric of the regionalized model will be returned as a
#             labeled pandas dataframe. Else, each metric will be returned as an
#             unlabeled numpy array.
#
#         Returns
#         -------
#         An instantiated RegionalizedModel object.
#         """
#         return cls(voxel_array.weights, voxel_array.nodes, *args, **kwargs)
#
#     def __init__(self, weights, nodes, source_key, target_key,
#                  ordering=None, dataframe=False):
#         if source_key.size != weights.shape[0]:
#             raise ValueError("rows of weights and elements in source_key "
#                              "must be equal size")
#         if target_key.size != nodes.shape[1]:
#             raise ValueError("columns of nodes and elements in target_key "
#                              "must be of equal size")
#
#         # metrics return dataframe?
#         self.weights = weights
#         self.nodes = nodes
#         self.source_key = source_key
#         self.target_key = target_key
#         self.ordering = ordering
#         self.dataframe = dataframe
#
#     def predict(self, X, normalize=False):
#         """Predict regional projection."""
#         # TODO : implement
#         raise NotImplementedError
#
#     def _regionalize_voxel_connectivity_array(self):
#         """Produces the full regionalized connectivity"""
#         # get counts for metrics
#         self.source_regions, self.source_counts = nonzero_unique(
#             self.source_key, return_counts=True)
#         self.target_regions, self.target_counts = nonzero_unique(
#             self.target_key, return_counts=True)
#
#         # integrate over target regions (array is 2x as wide)
#         temp = np.empty((self.target_regions.size, self.weights.shape[0]))
#         for i, region in enumerate(self.target_regions):
#
#             # same as voxel_array[:,cols].sum(axis=1), but more space efficient
#             columns = np.nonzero(self.target_key == region)[0]
#             temp[i, :] = self.weights.dot(np.sum(self.nodes[:, columns], axis=1))
#
#         # integrate over source regions
#         return unionize(temp, self.source_key).T
#
#     def _get_region_matrix(self):
#         region_matrix = self._regionalize_voxel_connectivity_array()
#
#         if self.ordering is not None:
#             order = lambda x: np.array(self.ordering)[np.isin(self.ordering, x)]
#             permutation = lambda x: np.argsort(np.argsort(order(x)))
#
#             source_perm = permutation(self.source_regions)
#             target_perm = permutation(self.target_regions)
#
#             self.source_regions = self.source_regions[source_perm]
#             self.target_regions = self.target_regions[target_perm]
#
#             self.source_counts = self.source_counts[source_perm]
#             self.target_counts = self.target_counts[target_perm]
#
#             region_matrix = region_matrix[np.ix_(source_perm, target_perm)]
#
#         if self.dataframe:
#             region_matrix = pd.DataFrame(region_matrix)
#             region_matrix.index = self.source_regions
#             region_matrix.columns = self.target_regions
#
#         return region_matrix
#
#     @property
#     def connection_strength(self):
#         """:math:`w_{ij}`
#
#            The sum of the voxel-scale connectivity between each pair
#            of source-target regions.
#         """
#         try:
#             return self._region_matrix
#         except AttributeError:
#             self._region_matrix = self._get_region_matrix()
#             return self._region_matrix
#
#     @property
#     def connection_density(self):
#         """:math:`w_{ij} / |Y|`
#
#           The average voxel-scale connectivity between each source
#           voxel to each source region.
#         """
#         return np.divide(self.connection_strength,
#                          self.target_counts[np.newaxis, :])
#
#     @property
#     def normalized_connection_strength(self):
#         """:math:`w_{ij} / |X|`
#
#           The average voxel-scale connectivity between each source
#           region to each target voxel"
#         """
#         return np.divide(self.connection_strength,
#                          self.source_counts[:, np.newaxis])
#
#     @property
#     def normalized_connection_density(self):
#         """:math:`w_{ij} / (|X| |Y|)`
#
#           The average voxel-scale connectivity between each pair of
#           source-target regions
#         """
#         return np.divide(self.connection_strength,
#                          np.outer(self.source_counts, self.target_counts))
