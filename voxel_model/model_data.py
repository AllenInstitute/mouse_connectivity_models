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

    # normalizations
    INJECTION_KEY="normalized_injection_density"
    PROJECTION_KEY="normalized_projection_density"

    # model experiment parameters
    MIN_INJECTION_VOLUME=0.0
    MIN_PROJECTION_VOLUME=0.0
    MIN_RATIO_CONTAINED_INJECTION=0.0
    
    def _test_experiment_parameters(self, exp):

        # defines if normalized
        injection = getattr(exp, INJECTION_KEY)
        projection = getattr(exp, PROJECTION_KEY)

        # masked
        masked_injection = self.source_mask.mask_volume(injection)
        masked_projection = self.target_mask.mask_volume(projection)

        masked_injection_volume = masked_injection.sum()
        masked_projection_volume = masked_projection.sum()
        contained_injection_ratios = masked_injection_volume / injection.sum()
        
        if all( (contained_injection_ratios >= self.MIN_RATIO_CONTAINED_INJECTION,
                 masked_injection_volume >= self.MIN_INJECTION_VOLUME,
                 masked_projection_volume >= self.MIN_PROJECTION_VOLUME) ):

            return masked_injection, masked_projection

        else:
            # caught implicitly in self._get_experiments()
            raise ValueError

    def _init_arrays(self):
        """ initializes arrays. """

        n = len(experiment_ids)
        annotation_dim = len(source_mask.annotation_shape)

        x = np.empty( (n, source_mask.masked_shape[0]), dtype=np.float32 )
        y = np.empty( (n, target_mask.masked_shape[0]), dtype=np.float32 )
        centroids = np.empty( (n, annotation_dim), dtype=np.float32 )

        return x, y, centroids

    def _get_experiments(self):
        """  ... """

        # initialize containers
        x, y, centroids = self._init_arrays()

        for i, experiment_id in enumerate(experiment_ids):

            # get experiment data
            exp = Experiment.from_mcc(mcc, experiment_id)

            try:
                # test if meets parameters
                injection, projection = self.test_experiment_parameters(exp)

            except ValueError:
                pass

            else:
                # update
                x[i] = injection
                y[i] = projection
                centroids[i] = exp.centroid

        # stack centroids, injections
        return np.hstack( (centroids, x) ), y

    def __init__(self, mcc, experiment_ids, source_mask, target_mask):
        self.mcc = mcc
        self.experiment_ids = experiment_ids
        self.source_mask = source_mask
        self.target_mask = target_mask

        # get all data & mask it to source/target masks
        self.X, self.y = self._get_experiments()

    @property
    def source_voxels(self):
        return self.source_mask.coordinates
