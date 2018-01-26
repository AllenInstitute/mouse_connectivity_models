# Authors:
# License:

from __future__ import absolute_import
import numpy as np

from collections import namedtuple
from .experiment import Experiment

def get_experiment_ids(mcc, structure_ids, cre=None):
    """Returns all experiment ids given some structure_ids
    PRIMARY INJECTION STRUCTURES
    """
    # filters injections by structure id OR DECENDENT
    experiments = mcc.get_experiments(dataframe=False, cre=cre,
                                      injection_structure_ids=structure_ids)
    return [ experiment['id'] for experiment in experiments ]

class ModelData(namedtuple("ModelData", ["X", "y", "source_voxels"])):
    """Container for model data...

    ...
    ...
    ...

    """
    __slots__ = ()

    # normalizations
    INJECTION_KEY="normalized_injection_density"
    PROJECTION_KEY="normalized_projection_density"

    # model experiment parameters
    MIN_INJECTION_SUM=0.0
    MIN_PROJECTION_SUM=0.0
    MIN_RATIO_CONTAINED_INJECTION=0.0

    def _valid_experiment(self, injection, projection, unmasked_injection_sum):

        inj_sum = injection.sum()
        proj_sum = projection.sum()
        inj_ratios = injection_sum / unmasked_injection_sum

        return all( (inj_ratios >= self.MIN_RATIO_CONTAINED_INJECTION,
                     inj_sum >= self.MIN_INJECTION_SUM,
                     proj_sum >= self.MIN_PROJECTION_SUM) )


    def _get_experiment_attrs(self, exp, source_mask, target_mask):

        # defines if normalized
        injection = source_mask.mask_volume( getattr(exp, self.INJECTION_KEY) )
        projection = target_mask.mask_volume( getattr(exp, self.PROJECTION_KEY) )

        unmasked_injection_sum = exp.injection_sum

        return injection, projection, unmasked_injection_sum

    @classmethod
    def from_mcc_and_masks(cls, mcc, source_mask, target_mask):
        """  ... """

        # for convenience
        structure_ids = source_mask.structure_ids
        hemisphere = source_mask.hemisphere

        # initialize containers
        x, y, centroids = [], [], []
        for experiment_id in get_experiment_ids(mcc, structure_ids):

            # get experiment data
            exp = Experiment.from_mcc( mcc, experiment_id,
                                       injection_hemisphere = hemisphere)

            # test if meets parameters
            inj, proj, unmasked_inj_sum = self._get_experiment_attrs(exp)

            if self._valid_experiment(inj, proj, unmasked_inj_sum):
                # update
                x.append( injection )
                y.append( projection )
                centroids.append( exp.centroid )

        # stack centroids, injections
        X = np.hstack( (np.asarray(centroids), np.asarray(x)) )

        return cls(X, np.asarray(y), source_mask.coordinates)


    def __new__(cls, X, y, source_voxels):

        if type(X) == np.ndarray and type(y) == np.ndarray:
            if X.shape[0] != y.shape[0]:
                raise ValueError("# of experiments in X and y is inconsistent")

        elif type(source_voxels) == np.ndarray:
            if source_voxels.shape[0] != injection_density.shape[1]:
                raise ValueError( "# of voxels in X and source_voxels "
                                  "is inconsistent" )
        else:
            raise ValueError( "X, y and source_voxels must all be of "
                              "type numpy.ndarray" )

        return super(ModelData, cls).__new__(X, y, source_voxels)
