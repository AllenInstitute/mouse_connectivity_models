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

def generate_experiments_from_mcc(mcc, experiment_ids,
                                  injection_hemisphere=None):
    """Generates Experiment objects.

    ...

    Parameters
    ----------
    """
    for eid in experiment_ids:
        yield Experiment.from_mcc( mcc, eid,
                                   injection_hemisphere=injection_hemisphere)

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

    @classmethod
    def _is_valid_experiment(cls, injection, projection, injection_ratio):
        """ ...

        Parameters
        ----------

        """

        return all( (injection_ratio >= cls.MIN_RATIO_CONTAINED_INJECTION,
                     injection.sum() >= cls.MIN_INJECTION_SUM,
                     projection.sum() >= cls.MIN_PROJECTION_SUM) )

    @classmethod
    def from_mcc_and_masks( cls, mcc, source_mask, target_mask,
            experiment_ids=None ):
        """  ... """

        # get list of experment ids
        if experiment_ids is None:
            experiment_ids = get_experiment_ids(mcc, source_mask.structure_ids)

        # initialize containers
        x, y, centroids = [], [], []
        for exp in generate_experiments_from_mcc( mcc, experiment_ids,
                                                  source_mask.hemisphere ):

            # get masked experiment data volumes
            injection = exp.mask_volume( cls.INJECTION_KEY, source_mask )
            projection = exp.mask_volume( cls.PROJECTION_KEY, target_mask )

            # get ratio in/out
            injection_ratio = exp.get_injection_ratio_contained( source_mask )

            if cls._is_valid_experiment( injection, projection, injection_ratio ):
                # update
                x.append( injection )
                y.append( projection )
                centroids.append( exp.centroid )

        # stack centroids, injections
        X = np.hstack( (np.asarray(centroids), np.asarray(x)) )

        return cls(X, np.asarray(y), source_mask.coordinates)


    def __new__(cls, X, y, source_voxels):

        if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            if X.shape[0] != y.shape[0]:
                raise ValueError("# of experiments in X and y is inconsistent")

        elif isinstance(source_voxels, np.ndarray):
            if source_voxels.shape[0] != injection_density.shape[1]:
                raise ValueError( "# of voxels in X and source_voxels "
                                  "is inconsistent" )
        else:
            raise ValueError( "X, y and source_voxels must all be of "
                              "type numpy.ndarray" )

        return super(ModelData, cls).__new__(cls, X, y, source_voxels)
