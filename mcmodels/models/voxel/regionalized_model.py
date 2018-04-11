"""
Module containing the RegionalizedModel object, used in evaluating the
voxel-voxel model at the level of regions.
"""

# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: BSD 3

from __future__ import division, absolute_import

import numpy as np
import pandas as pd

from ...utils import nonzero_unique, unionize


class RegionalizedModel(object):
    """Regionalization/Parcelation of VoxelModel.

    Regionalizes the connectivity model in VoxelModel given a brain parcelation.
            Metric with which to represent the regionalized connectivity.
            Valid choices are:
            - "connection_strength" (default)

               ..math:: W = w_ij \|X\|\|Y\|

               The sum of the voxel-scale connectivity between each pair
               of source-target regions.

            - "connection_density"

              ..math:: W = w_ij \|X\|

              The average voxel-scale connectivity between each source
              voxel to each source region.

            - "normalized_connection_strength"
              ..math:: W = w_ij \|Y\|

              The average voxel-scale connectivity between each source
              region to each target voxel"

            - "normalized_connection_density"
              ..math:: W = w_ij

              The average voxel-scale connectivity between each pair of
              source-target regions

    Parameters
    ----------

    source_key : array-like, shape=(n_source_voxels,)
        Flattened key relating each source voxel to a given brain region.

    target_key : array-like, shape=(n_target_voxels,)
        Flattened key relating each target voxel to a given brain region.

    Examples
    --------
    """
    VALID_REGION_METRICS = [
        "connection_strength",
        "connection_density",
        "normalized_connection_strength",
        "normalized_connection_density"
    ]

    @classmethod
    def from_voxel_array(cls, voxel_array, *args, **kwargs):
        """If weights and nodes passed explicitly"""
        return cls(voxel_array.weights, voxel_array.nodes, *args, **kwargs)

    def __init__(self, weights, nodes, source_key, target_key,
                 ordering=None, dataframe=False):
        if source_key.size != weights.shape[0]:
            raise ValueError("rows of weights and elements in source_key "
                             "must be equal size")
        if target_key.size != nodes.shape[1]:
            raise ValueError("columns of nodes and elements in target_key "
                             "must be of equal size")

        # metrics return dataframe?
        self.weights = weights
        self.nodes = nodes
        self.source_key = source_key
        self.target_key = target_key
        self.ordering = ordering
        self.dataframe = dataframe

    def predict(self, X, normalize=False):
        """Predict regional projection."""
        # TODO : implement
        raise NotImplementedError

    def _regionalize_voxel_connectivity_array(self):
        """Produces the full regionalized connectivity"""
        # get counts for metrics
        self.source_regions, self.source_counts = nonzero_unique(
            self.source_key, return_counts=True)
        self.target_regions, self.target_counts = nonzero_unique(
            self.target_key, return_counts=True)

        # integrate over target regions (array is 2x as wide)
        temp = np.empty((self.target_regions.size, self.weights.shape[0]))
        for i, region in enumerate(self.target_regions):

            # same as voxel_array[:,cols].sum(axis=1), but more space efficient
            columns = np.nonzero(self.target_key == region)[0]
            temp[i, :] = self.weights.dot(np.sum(self.nodes[:, columns], axis=1))

        # integrate over source regions
        return unionize(temp, self.source_key).T

    def _get_region_matrix(self):
        region_matrix = self._regionalize_voxel_connectivity_array()

        if self.ordering is not None:
            order = lambda x: np.array(self.ordering)[np.isin(self.ordering, x)]
            permutation = lambda x: np.argsort(np.argsort(order(x)))

            source_perm = permutation(self.source_regions)
            target_perm = permutation(self.target_regions)

            self.source_regions = self.source_regions[source_perm]
            self.target_regions = self.target_regions[target_perm]

            self.source_counts = self.source_counts[source_perm]
            self.target_counts = self.target_counts[target_perm]

            region_matrix = region_matrix[np.ix_(source_perm, target_perm)]

        if self.dataframe:
            region_matrix = pd.DataFrame(region_matrix)
            region_matrix.index = self.source_regions
            region_matrix.columns = self.target_regions

        return region_matrix

    @property
    def connection_strength(self):
        """..math:: w_ij \|X\|\|Y\|"""
        try:
            return self._region_matrix
        except AttributeError:
            self._region_matrix = self._get_region_matrix()
            return self._region_matrix

    @property
    def connection_density(self):
        """..math:: w_ij \|X\|"""
        return np.divide(self.connection_strength,
                         self.target_counts[np.newaxis, :])

    @property
    def normalized_connection_strength(self):
        """..math:: w_ij \|Y\|"""
        return np.divide(self.connection_strength,
                         self.source_counts[:, np.newaxis])

    @property
    def normalized_connection_density(self):
        """..math:: w_ij"""
        return np.divide(self.connection_strength,
                         np.outer(self.source_counts, self.target_counts))
