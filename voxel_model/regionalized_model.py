"""
Module containing the RegionalizedModel object, used in evaluating the
voxel-voxel model at the level of regions.
"""

# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import division, absolute_import
import numpy as np

from .utils import lex_ordered_unique


class RegionalizedModel(object):
    """Regionalization/Parcelation of VoxelModel.

    Regionalizes the connectivity model in VoxelModel given a brain parcelation.
            Metric with which to represent the regionalized connectivity.
            Valid choices are:
                * "connection_strength" (default)
                    W = w_ij |X||Y|
                    The sum of the voxel-scale connectivity between each pair
                    of source-target regions.

                * "connection_density"
                    W = w_ij |X|
                    The average voxel-scale connectivity between each source
                    voxel to each source region.

                * "normalized_connection_strength"
                    W = w_ij |Y|
                    The average voxel-scale connectivity between each source
                    region to each target voxel"

                * "normalized_connection_density"
                    W = w_ij
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

    def __init__(self, weights, nodes, source_key, target_key, ordering=None):
        if source_key.size != weights.shape[0]:
            raise ValueError("rows of weights and elements in source_key "
                             "must be equal size")

        if target_key.size != nodes.shape[1]:
            raise ValueError("columns of nodes and elements in target_key "
                             "must be of equal size")

        # want only valid indices (source/target keys likely to have zeros)
        rows = source_key.nonzero()[0]
        cols = target_key.nonzero()[0]

        # subset
        self.weights = weights[rows, :]
        self.nodes = nodes[:, cols]
        self.source_key = source_key[rows]
        self.target_key = target_key[cols]
        self.ordering = ordering

    def predict(self, X, normalize=False):
        """Predict regional projection."""
        # TODO : implement
        raise NotImplementedError

    def _get_unique_counts(self, key):
        """Returns unique and counts in appropriate order."""
        if self.ordering is None:
            return np.unique(key, return_counts=True)

        return lex_ordered_unique(key, self.ordering, allow_extra=True,
                                  return_counts=True)

    def _get_region_matrix(self):
        """Produces the full regionalized connectivity"""
        # get counts
        source_regions, self.source_counts = self._get_unique_counts(self.source_key)
        target_regions, self.target_counts = self._get_unique_counts(self.target_key)

        # integrate target regions
        temp = np.empty((target_regions.size, self.weights.shape[0]))
        for i, region in enumerate(target_regions):

            # same as voxel_array[:,cols].sum(axis=1), but more efficient
            columns = np.nonzero(self.target_key == region)[0]
            temp[i, :] = self.weights.dot(np.sum(self.nodes[:, columns], axis=1))

        # integrate source regions
        region_matrix = np.empty((source_regions.size, target_regions.size))
        for i, region in enumerate(source_regions):

            # NOTE : if region were 1 voxel, would not work?
            columns = np.nonzero(self.source_key == region)[0]
            region_matrix[i, :] = temp[:, columns].sum(axis=1)

        return region_matrix

    @property
    def connection_strength(self):
        """w_ij |X||Y|"""
        try:
            return self._region_matrix
        except AttributeError:
            self._region_matrix = self._get_region_matrix()
            return self._region_matrix

    @property
    def connection_density(self):
        """w_ij |X|"""
        return np.divide(self.connection_strength,
                         self.target_counts[np.newaxis, :])

    @property
    def normalized_connection_strength(self):
        """w_ij |Y|"""
        return np.divide(self.connection_strength,
                         self.source_counts[:, np.newaxis])

    @property
    def normalized_connection_density(self):
        """w_ij"""
        return np.divide(self.connection_strength,
                         np.outer(self.source_counts, self.target_counts))
