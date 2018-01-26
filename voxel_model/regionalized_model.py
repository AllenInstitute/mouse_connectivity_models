from __future__ import division, absolute_import
import numpy as np

from .utils import lex_ordered_unique_counts

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
        # valid indices (source/target keys likely to have zeros)
        rows = source_key.nonzero()[0]
        cols = target_key.nonzero()[0]

        # subset
        self.weights = weights[rows, :]
        self.nodes = nodes[:, cols]
        self.source_key = source_key[ rows ]
        self.target_key = target_key[ cols ]
        self.ordering = ordering

    def predict(self, X, normalize=False):
        raise NotImplementedError

    def _get_unique_counts(self, keyname):
        """ ... """
        key = getattr(self, keyname)

        if self.ordering is not None:
            return lex_ordered_unique_counts( key, self.ordering )
        else:
            return np.unique( key, return_counts=True)

    def _get_region_matrix(self):
        """Produces the full regionalized connectivity"""

        # get counts
        source_regions, self.source_counts = self._get_unique_counts("source_key")
        target_regions, self.target_counts = self._get_unique_counts("target_key")

        # integrate target regions
        # NOTE: probably more efficient to sort then stride by nt_regions
        temp = np.empty( (target_regions.size, self.weights.shape[0]) )
        for i, region in enumerate(target_regions):
            cols = np.isin(self.target_key, region)

            # same output as weights.dot(nodes[:,cols]).sum(axis=1)
            # but much more memory efficient to compute sum first
            temp[i,:] = self.weights.dot(
                np.einsum('ji->j', self.nodes[:,cols])
            )

        # integrate source regions
        # NOTE: probably more efficient to sort then stride by ns_regions
        region_matrix = np.empty( (source_regions.size, target_regions.size) )
        for i, region in enumerate(source_regions):
            cols = np.isin(self.source_key, region)

            # NOTE : if region were 1 voxel, would not work
            region_matrix[i,:] = temp[:,cols].sum(axis=1)

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
