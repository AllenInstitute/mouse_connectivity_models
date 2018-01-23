# Authors:
# License:

from __future__ import absolute_import
import numpy as np

from .experiment import Experiment

class ModelData(object):
    """Container for model data...

    ...
    ...
    ...

    """
    def _get_experiments(self):
        """  ... """
        # masked columns
        source_idx = self.source_mask.nonzero
        target_idx = self.target_mask.nonzero

        X, y, centroids, total_volumes = [], [], [], []
        for experiment_id in self.experiment_ids:
            # get experiment data
            exp = Experiment(self.mcc, experiment_id)

            # for min_ratio_contained
            total_volumes.append( exp.normalized_injection_density.sum() )

            # update
            X.append( exp.normalized_injection_density[source_idx] )
            y.append( exp.normalized_projection_density[target_idx] )
            centroids.append( exp.centroid )

        # stack centroids, injections
        X = np.hstack( (np.asarray(centroids), np.asarray(X)) )

        # return arrays
        return X, np.asarray(y), np.asarray(total_volumes)

    def __init__(self, mcc, experiment_ids, source_mask, target_mask,
                 min_injection_volume=0.0, min_projection_volume=0.0,
                 min_ratio_contained=0.0):
        self.mcc = mcc
        self.experiment_ids = experiment_ids
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.min_injection_volume = min_injection_volume
        self.min_projection_volume = min_projection_volume
        self.min_ratio_contained = min_ratio_contained

        # get all data
        self._X, self._y, self._total_volumes = self._get_experiments()

    def _get_valid_rows(self):
        """ ... """
        # injection volumes
        masked_volumes = self._X.sum(axis=1)
        contained_ratios = np.divide(masked_volumes, self._total_volumes)

        # tests
        valid_inj_ratios = contained_ratios >= self.min_ratio_contained
        valid_injections = masked_volumes >= self.min_injection_volume
        valid_projections = self._y.sum(axis=1) >= self.min_projection_volume

        # return valid rows
        valid = ( valid_inj_ratios, valid_injections, valid_projections )
        return np.logical_and.reduce(valid)

    @property
    def valid_rows(self):
        try:
            return self._valid_rows
        except AttributeError:
            self._valid_rows = self._get_valid_rows()
            return self._valid_rows

    @property
    def valid_experiment_ids(self):
        return self.experiment_ids[self.valid_rows]

    @property
    def X(self):
        return self._X[self.valid_rows]

    @property
    def y(self):
        return self._y[self.valid_rows]

    @property
    def source_voxels(self):
        return self.source_mask.coordinates
