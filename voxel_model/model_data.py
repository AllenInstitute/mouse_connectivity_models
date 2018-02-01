"""Module containing ModelData container"""

# TODO : revise ModelData docstring :: Attributes
# TODO : revise ModelData docstring :: Examples

# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import absolute_import
from collections import namedtuple
import numpy as np

from .experiment import Experiment

def get_experiment_ids(mcc, structure_ids, cre=None):
    """Returns all experiment ids given some structure_ids
    """
    # filters injections by structure id OR DECENDENT
    experiments = mcc.get_experiments(dataframe=False, cre=cre,
                                      injection_structure_ids=structure_ids)
    return [experiment['id'] for experiment in experiments]

def generate_experiments_from_mcc(mcc, experiment_ids,
                                  injection_hemisphere=None):
    """Generates Experiment objects.

    Utilizes the MouseConnectivityCache object from
    allensdk.mcc.core.mouse_connectivity_cache to pull grid data and generate
    Experiment objects (see voxel_model.experiment).

    Parameters
    ----------
    mcc : MouseConnectivityCache instance.
        Object used to pull grid data.
        see allensdk.core.mouse_connectivity_cache module for more info.

    experiment_ids : list
        List of experiment ids to include as potential experiments.

    injection_hemisphere : int, optional (default=None)
        If None, defualts to Experiment.DEFAULT_INJECTION_HEMISPHERE (2).
        Valid arguments are:
            1 : left hemisphere
            2 : right hemisphere
            3 : both hemispheres

    Yields
    ------
    Experiment object.

    """
    for eid in experiment_ids:
        yield Experiment.from_mcc(mcc, eid,
                                  injection_hemisphere=injection_hemisphere)

class ModelData(namedtuple("ModelData", ["centroids", "injections",
                                       "projections", "source_voxels"])):
    """Container for grid data.

    A conveinence class used to generate arrays used in the model defined in
    voxel_model.regressors. Use class method from_mcc_and_masks to pull
    experiment grid data using the allensdk package.

    see voxel_model.experiment for additional information on attributes.

    Attributes
    ----------
    centroids : array, shape (n_experiments, volume_dim)
        Centroids of the injection densities in each experiment.

    injections : array, shape (n_experiments, n_source_voxels)
        Injection density for each experiment, (normalized?).

    projections : array, shape (n_experiments, n_target_voxels)
        Projection density for each experiment, normalized by the sum
        of the injection density for each experiment.

    source_voxels : array, shape (n_source_voxels, volume_dim)
        Grid coordinates of the source voxels mapping to the columns of
        injections.

    Examples
    --------
    >>> from voxel_model.model_data import ModelData

    """
    # NOTE : verbatim from sklearn.gaussian_process.kernels
    # raw namedtuple is very memory efficient as it packs the attributes
    # in a struct to get rid of the __dict__ of attributes in particular it
    # does not copy the string for the keys on each instance.
    # By deriving a namedtuple class just to introduce the __init__ method we
    # would also reintroduce the __dict__ on the instance. By telling the
    # Python interpreter that this subclass uses static __slots__ instead of
    # dynamic attributes. Furthermore we don't need any additional slot in the
    # subclass so we set __slots__ to the empty tuple

    __slots__ = ()

    # normalizations
    INJECTION_KEY = "normalized_injection_density"
    PROJECTION_KEY = "normalized_projection_density"

    # model experiment parameters
    MIN_INJECTION_SUM = 0.0
    MIN_PROJECTION_SUM = 0.0
    MIN_RATIO_CONTAINED_INJECTION = 0.0

    @classmethod
    def _is_valid_experiment(cls, injection, projection, injection_ratio):
        """Returns if given experiment meets requirements."""
        return all((injection_ratio >= cls.MIN_RATIO_CONTAINED_INJECTION,
                    injection.sum() >= cls.MIN_INJECTION_SUM,
                    projection.sum() >= cls.MIN_PROJECTION_SUM))

    @classmethod
    def from_mcc_and_masks(cls, mcc, source_mask, target_mask,
                           experiment_ids=None):
        """Alternative constructor allowing for pulling grid data.

        Uses voxel_model.experment to generate grid data using allensdk.
        Masks are supplied to restrict the grid data to only those brain
        regions that are of interest.

        see voxel_model.experment and voxel_model.masks for more info.

        Parameters
        ----------
        mcc : MouseConnectivityCache instance.
            Object used to pull grid data.
            see allensdk.core.mouse_connectivity_cache module for more info.

        source_mask : Mask instance
            Object used to mask injection grid data as well as define relevant
            experiments and source_voxels.

        target_mask : Mask instance
            Object used to mask projection grid data.

        experiment_ids : list
            List of experiment ids to include as potential experiments.

        """
        # get list of experment ids
        if experiment_ids is None:
            experiment_ids = get_experiment_ids(mcc, source_mask.structure_ids)

        # initialize containers
        centroids, injections, projections = [], [], []
        for exp in generate_experiments_from_mcc(mcc, experiment_ids,
                                                 source_mask.hemisphere):

            # get masked experiment data volumes
            injection = exp.mask_volume(cls.INJECTION_KEY, source_mask)
            projection = exp.mask_volume(cls.PROJECTION_KEY, target_mask)

            # get ratio in/out
            injection_ratio = exp.get_injection_ratio_contained(source_mask)

            if cls._is_valid_experiment(injection, projection, injection_ratio):
                # update
                injections.append(injection)
                projections.append(projection)
                centroids.append(exp.centroid)

        return cls(np.asarray(centroids), np.asarray(injections),
                   np.asarray(projections), source_mask.coordinates)


    def __new__(cls, centroids, injections, projections, source_voxels):
        # assumes all are numpy arrays inheritly
        if injections.shape[0] != projections.shape[0]:
            raise ValueError("# of experiments in injections and "
                             "projections is inconsistent")

        if injections.shape[0] != centroids.shape[0]:
            raise ValueError("# of experiments in centroids and "
                             "injections/projections is inconsistent")

        if source_voxels.shape[0] != injections.shape[1]:
            raise ValueError("# of voxels in injections and source_voxels "
                             "is inconsistent")

        if centroids.shape[1] != source_voxels.shape[1]:
            raise ValueError("dimension of centroids and voxels is not equal")

        return super(ModelData, cls).__new__(cls, centroids, injections,
                                             projections, source_voxels)
