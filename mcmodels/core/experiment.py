"""
Module containing Experiment object and supporting functions
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

from __future__ import division
from functools import partial

import numpy as np

from .utils import compute_centroid, get_injection_hemisphere_id


def _pull_grid_data(voxel_model_cache, experiment_id):
    """Pulls data volumes using VoxelModelCache object.

    Parameters
    ----------
    voxel_model_cache : VoxelModelCache instance.
        Object used to pull grid data.

    experiment_id : int
        Experiment id of the experiment from which to pull grid data.

    Returns
    -------
    dict
        Container of relevant data volumes.
        See allensdk.core.mouse_connectivity_cache for description of volumes.

    Notes
    -----
    voxel_model_cache.get_<data_volume> returns a tuple (data_volume, meta_data).
    We only care about the data volume.

    """
    return {
        "data_mask" : voxel_model_cache.get_data_mask(experiment_id)[0],
        "injection_density" : voxel_model_cache.get_injection_density(experiment_id)[0],
        "injection_fraction" : voxel_model_cache.get_injection_fraction(experiment_id)[0],
        "projection_density" : voxel_model_cache.get_projection_density(experiment_id)[0]
    }


def _mask_data_volume(data_volume, data_mask, tolerance=0.):
    """Masks a given data volume in place.

    Parameters
    ----------
    data_volume : array, shape (x_ccf, y_ccf, z_ccf)
        Data volume to be masked.

    data_mask : array, shape (x_ccf, y_ccf, z_ccf)
        data_mask for given experiment (values in [0,1])
        See allensdk.core.mouse_connectivity_cache for more info.

    tolerance : float, optional (default=0.0)
        tolerance with which to define bad voxels in data_mask.


    Returns
    -------
    data_volume
        data_volume parameter masked in place.

    """
    if data_volume.shape != data_mask.shape:
        raise ValueError("data_volume and data_mask must have same shape.")

    # mask data volume
    data_volume[data_mask < tolerance] = 0.

    return data_volume


def _compute_true_injection_density(injection_density, injection_fraction, inplace=False):
    """Computes 'true' injecton_density.

    Takes into consideration injection fracion (proportion of pixels in the
    annotated injection site).

    Parameters
    ----------
    injection_density : array, shape (x_ccf, y_ccf, z_ccf)
        injection_density data volume.

    injection_fraction : array, shape (x_ccf, y_ccf, z_ccf)
        injection_fraction data volume.

    inplace : boolean
        If True, overwrites injection_density parameter, else returns new array.

    Returns
    -------
    array, shape (x_ccf, y_ccf, z_ccf)
        'true' injection density : injection_density * injection_fraction
    """
    if injection_density.shape != injection_fraction.shape:
        raise ValueError("injection_density and injection_fraction must "
                         "have same shape.")

    if inplace:
        np.multiply(injection_density, injection_fraction, injection_density)
        return injection_density

    return np.multiply(injection_density, injection_fraction)


class Experiment(object):
    """Class containing the data from an anterograde injection

    Experiment conveniently compiles the relevant information from a given
    anterograde viral tracing experiment data.

    Parameters
    ----------
    voxel_model_cache : VoxelModelCache object
        This supplies the interface for pulling experimental data.

    experiment_id : int
        AllenSDK id assigned to given experiment

    Examples
    --------
    >>> from voxel_model.core import Experiment
    >>> from voxel_model import VoxelModelCache
    >>> voxel_model_cache = VoxelModelCache(resolution=100)
    >>> eid = 100141273
    >>> exp = Experiment(voxel_model_cache, eid)
    >>> exp.injection_density.shape
    (132,80,114)

    """
    DEFAULT_DATA_MASK_TOLERANCE = 0.5

    @classmethod
    def from_voxel_model_cache(cls, voxel_model_cache, experiment_id,
                               data_mask_tolerance=None):
        """Alternative constructor allowing for pulling grid data.

        Parameters
        ----------
        voxel_model_cache : VoxelModelCache instance.
            Object used to pull grid data.

        experiment_id : int
            Experiment id of the experiment from which to pull grid data.

        injection_hemisphere : int, optional (default=None)
            If None, defualts to Experiment.DEFAULT_INJECTION_HEMISPHERE (2).
            Valid arguments are:
            - 1 : left hemisphere
            - 2 : right hemisphere
            - 3 : both hemispheres
        """
        if data_mask_tolerance is None:
            data_mask_tolerance = cls.DEFAULT_DATA_MASK_TOLERANCE

        # pull data
        data_volumes = _pull_grid_data(voxel_model_cache, experiment_id)

        # compute 'true' injection density (inplace)
        _compute_true_injection_density(data_volumes["injection_density"],
                                        data_volumes["injection_fraction"],
                                        inplace=True)

        # mask data in place
        mask_func = partial(_mask_data_volume,
                            data_mask=data_volumes["data_mask"],
                            tolerance=data_mask_tolerance)

        injection_density = mask_func(data_volumes["injection_density"])
        projection_density = mask_func(data_volumes["projection_density"])

        return cls(injection_density, projection_density)


    def __init__(self, injection_density=None, projection_density=None):
        # assume numpy array
        if injection_density.shape != projection_density.shape:
            raise ValueError("injection_density and projection_density "
                             "must be the same shape!")

        self.injection_density = injection_density
        self.projection_density = projection_density

    @property
    def injection_hemisphere_id(self):
        """Returns injection hemisphere"""
        return get_injection_hemisphere_id(self.injection_density, majority=True)

    @property
    def bilateral_injection(self):
        """Returns injection hemisphere"""
        return get_injection_hemisphere_id(self.injection_density) == 3

    @property
    def injection_volume(self):
        """Returns total injection volume = sum(injection_density)"""
        return self.injection_density.sum()

    @property
    def projection_volume(self):
        """Returns total projection volume = sum(projection_density)"""
        return self.projection_density.sum()

    @property
    def centroid(self):
        """Returns centroid of the injection density."""
        return compute_centroid(self.injection_density)

    @property
    def normalized_injection_density(self):
        """Returns injection density normalized to have unit sum."""
        return self.injection_density / self.injection_volume

    @property
    def normalized_projection_density(self):
        """Returns projection_density normalized by the total injection_density"""
        return self.projection_density / self.injection_volume

    def get_injection(self, normalized=False):
        if normalized:
            return self.normalized_injection_density
        return self.injection_density

    def get_projection(self, normalized=False):
        if normalized:
            return self.normalized_projection_density
        return self.projection_density

    def flip(self):
        """Reflects experiment along midline.

        Returns
        -------
        self - flipped experiment
        """
        self.injection_density = self.injection_density[..., ::-1]
        self.projection_density = self.projection_density[..., ::-1]

        return self
