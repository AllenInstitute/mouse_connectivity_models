from __future__ import division, absolute_import
import numpy as np

from .utils import lex_ordered_unique_counts

__all__ = [
    "RegionalizedModel"
]

def _generate_column_sets(key, region_set):
    """Yields indices of columns where kye==region...

    ...

    Parameters
    ----------
    """
    for region in region_set:
        yield np.isin(key, region).nonzero()[0]

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

    def __init__(self, weights, nodes, source_key, target_key, ordering=None):
        if weights.shape[1] != nodes.shape[0]:
            raise ValueError("weights and nodes must have equal inner dimension")

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
        # TODO : look into if copy necessary/better performance
        self.weights = weights[rows, :].copy()
        self.nodes = nodes[:, cols].copy()
        self.source_key = source_key[ rows ].copy()
        self.target_key = target_key[ cols ].copy()
        self.ordering = ordering

    def predict(self, X, normalize=False):
        raise NotImplementedError

    def _get_unique_counts(self, key):
        """ ... """
        if self.ordering is not None:
            return lex_ordered_unique_counts( key, self.ordering )
        else:
            return np.unique( key, return_counts=True)

    def _get_region_matrix(self):
        """Produces the full regionalized connectivity"""

        # get counts
        source_regions, source_counts = self._get_unique_counts(self.source_key)
        target_regions, target_counts = self._get_unique_counts(self.target_key)

        # integrate target regions
        temp = np.empty( (target_regions.size, self.weights.shape[0]) )
        column_iterator = _generate_column_sets(self.target_key, target_regions)

        for i, columns in enumerate(column_iterator):
            # same output as weights.dot(nodes[:,cols]).sum(axis=1)
            # but much more efficient to compute sum first
            temp[i,:] = self.weights.dot( self.nodes[:,columns].sum(axis=1) )

        # integrate source regions
        region_matrix = np.empty( (source_regions.size, temp.size[0]) )
        column_iterator = _generate_column_sets(self.source_key, source_regions)

        for i, columns in enumerate(column_iterator):
            # NOTE : if region were 1 voxel, would not work
            region_matrix[i,:] = temp[:,columns].sum(axis=1)

        # want counts for metrics
        self.source_counts = source_counts
        self.target_counts = target_counts
        return region_matrix

    @property
    def region_matrix(self):
        try:
            return self._region_matrix
        except AttributeError:
            self._region_matrix = self._get_region_matrix()
            return self._region_matrix

    def get_metric(self, metric):
        """ ... """
        if metric == "connection_strength":
            # w_ij |X||Y|
            return self.region_matrix

        elif metric == "connection_density":
            # w_ij |X|
            return np.divide(self.region_matrix,
                             self.target_counts[np.newaxis,:])

        elif metric == "normalized_connection_strength":
            # w_ij |Y|
            return np.divide(self.region_matrix,
                             self.source_counts[:,np.newaxis])

        elif metric == "normalized_connection_density":
            # w_ij
            return np.divide(self.region_matrix,
                             np.outer(self.source_counts, self.target_counts))
        else:
            raise ValueError("metric must be one of", self.VALID_REGION_METRICS)
