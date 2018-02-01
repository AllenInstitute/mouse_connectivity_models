"""Module containing Experiment object and supporting functions"""

# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import division, absolute_import
from functools import partial
import numpy as np

from .masks import Mask

__all__ = ["Experiment", "compute_centroid"]

def _pull_data_volumes(mcc, experiment_id):
    """Pulls data volumes using MouseConnectivityCahce object.

    Parameters
    ----------
    mcc : MouseConnectivityCache instance.
        Object used to pull grid data.
        see allensdk.core.mouse_connectivity_cache module for more info.

    experiment_id : int
        Experiment id of the experiment from which to pull grid data.

    Returns
    -------
    dict
        Container of relevant data volumes.
        See allensdk.core.mouse_connectivity_cache for description of volumes.

    Notes
    -----
    mcc.get_<data_volume> returns a tuple ( data_volume, meta_data ). We only
    care about the data volume.

    """
    return {
        "data_mask" : mcc.get_data_mask(experiment_id)[0],
        "injection_density" : mcc.get_injection_density(experiment_id)[0],
        "injection_fraction" : mcc.get_injection_fraction(experiment_id)[0],
        "projection_density" : mcc.get_projection_density(experiment_id)[0]
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

def _compute_true_injection_density(injection_density, injection_fraction,
                                    inplace=False):
    """Computes 'true' injecton_density.

    Takes into consideration injection fracion (proportion of pixels in the
    annotated injection site).

        see allensdk.core.mouse_connectivity_cache module for more info.

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

def _get_injection_hemisphere(injection_density):
    """Gets injection hemisphere based on injection density.

    Defines injection hemisphere by the ratio of the total injection_density
    in each hemisphere.

    Parameters
    ----------
    injection_density : array, shape (x_ccf, y_ccf, z_ccf)
        injection_density data volume.

    Returns
    -------
    int
        injection_hemisphere
    """
    if len(injection_density.shape) != 3:
        raise ValueError("injection_density must be 3-array")

    # split along depth dimension (forces arr.shape[2] % 2 == 0)
    l_hemi, r_hemi = np.split(injection_density, 2, axis=2)

    # return injecton hemisphere based on sum of injection_density
    if l_hemi.sum() > r_hemi.sum():
        return 1

    return 2

def _flip_hemisphere(data_volume):
    """Flips data volume along 2 axis (hemipshere).

    Parameters
    ----------
    data_volume : array, shape (x_ccf, y_ccf, z_ccf)
        data volume.

    Returns
    -------
    flipped data_volume along last axis.

    """
    if len(data_volume.shape) != 3:
        raise ValueError("Must be 3-array")

    return data_volume[..., ::-1]

def compute_centroid(injection_density):
    """Computes centroid in index coordinates.

    Parameters
    ----------
    injection_density : array, shape (x_ccf, y_ccf, z_ccf)
        injection_density data volume.

    Returns
    -------
        centroid onf injection_density in index coordinates.

    """
    nonzero = injection_density[ injection_density.nonzero() ]
    voxels = np.argwhere( injection_density )

    return np.dot(nonzero, voxels) / injection_density.sum()

class Experiment(object):
    """Class containing the data from an anterograde injection

    Experiment conveniently compiles the relevant information from a given
    anterograde viral tracing experiment pulled from the AllenSDK
    MouseConnectivityCache module.

    See allensdk.core.mouse_connectivity_cache for more information.

    Parameters
    ----------
    mcc : allensdk.core.mouse_connectivity_cache.MouseConnectivityCache object
        This supplies the interface for pulling experimental data
        from the AllenSDK.

    experiment_id : int
        AllenSDK id assigned to given experiment

    Examples
    --------
    >>> from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    >>> from voxel_model.experiment import Experiment
    >>> mcc = MouseConnectivityCache(resolution=100)
    >>> eid = 100141273
    >>> exp = Experiment(mcc, eid)
    >>> exp.injection_density.shape
    (132,80,114)

    """
    DATA_MASK_TOLERANCE = 0.5
    DEFAULT_INJECTION_HEMISPHERE = 2
    VALID_INJECTION_HEMISPHERES = [1, 2, 3]

    @classmethod
    def from_mcc(cls, mcc, experiment_id, injection_hemisphere=None):
        """Alternative constructor allowing for pulling grid data.

        see allensdk.core.mouse_connectivity_cache module for more info.

        Parameters
        ----------
        mcc : MouseConnectivityCache instance.
            Object used to pull grid data.
            see allensdk.core.mouse_connectivity_cache module for more info.

        experiment_id : int
            Experiment id of the experiment from which to pull grid data.

        injection_hemisphere : int, optional (default=None)
            If None, defualts to Experiment.DEFAULT_INJECTION_HEMISPHERE (2).
            Valid arguments are:
                1 : left hemisphere
                2 : right hemisphere
                3 : both hemispheres

        """
        if injection_hemisphere is None:
            injection_hemisphere = cls.DEFAULT_INJECTION_HEMISPHERE

        elif injection_hemisphere not in cls.VALID_INJECTION_HEMISPHERES:
            raise ValueError("Injection hemisphere must be in "
                             "{.VALID_INJECTION_HEMISPHERES}".format(cls))

        # pull data
        data_volumes = _pull_data_volumes(mcc, experiment_id)

        # compute 'true' injection density (inplace)
        _compute_true_injection_density(data_volumes["injection_density"],
                                        data_volumes["injection_fraction"],
                                        inplace=True)

        # mask data in place
        mask_func = partial(_mask_data_volume,
                            data_mask=data_volumes["data_mask"],
                            tolerance=cls.DATA_MASK_TOLERANCE)

        injection_density = mask_func(data_volumes["injection_density"])
        projection_density = mask_func(data_volumes["projection_density"])

        # check injection hemisphere
        computed_hemisphere = _get_injection_hemisphere(injection_density)
        if computed_hemisphere != injection_hemisphere:

            # flip experiment
            injection_density, projection_density = map(_flip_hemisphere,
                                                        (injection_density,
                                                         projection_density))

        return cls(injection_density, projection_density)


    def __init__(self, injection_density=None, projection_density=None):
        # assume numpy array
        if injection_density.shape != projection_density.shape:
            raise ValueError("injection_density and projection_density "
                             "must be the same shape!")

        self.injection_density = injection_density
        self.projection_density = projection_density

    @property
    def centroid(self):
        return compute_centroid(self.injection_density)

    @property
    def normalized_injection_density(self):
        return self.injection_density / self.injection_density.sum()

    @property
    def normalized_projection_density(self):
        return self.projection_density / self.injection_density.sum()

    def get_injection_ratio_contained(self, mask):
        """Returns the raito contained in a given mask.

        ...

        Parameters
        ----------
        """
        if isinstance(mask, Mask):
            masked_injection = self.mask_volume( "injection_density", mask )
        else:
            # assume np.ndarray
            if mask.shape != self.injection_density.shape:
                raise ValueError( "if mask is array, it must have the "
                                  "same shape as injection density" )

            masked_injection = self.injection_density[ mask.nonzero() ]

        return masked_injection.sum() / self.injection_density.sum()

    def mask_volume(self, volume, mask):
        """Returns masked volume (flattened)

        ...

        Parameters
        ----------
        """
        try:
            # get volume
            data_volume = getattr(self, volume)
        except AttributeError:
            raise ValueError( "volume must be a valid data_volume" )

        if isinstance(mask, Mask):
            return mask.mask_volume( data_volume )
        else:
            # assume np.ndarray
            if mask.shape != self.injection_density.shape:
                raise ValueError( "if mask is array, it must have the "
                                  "same shape as injection density" )

            return data_volume[ mask.nonzero() ]
