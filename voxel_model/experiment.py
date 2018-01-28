# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import division
import numpy as np

__all__ = ["Experiment"]

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

    Attributes
    ----------
    data_mask : array-like, shape=(x_ccf, y_ccf, z_ccf)
        Mask of invalid voxels.

    injection_density : array-like, shape=(x_ccf, y_ccf, z_ccf)
        Volume in which values correspond to segmented viral injection density.

    injection_fraction : array-like, shape=(x_ccf, y_ccf, z_ccf)
        Volume in which values correspond to segmented viral injection fraction.
        In other words, the fraction of the voxel that lies within the annotation.

    projection_density : array-like, shape=(x_ccf, y_ccf, z_ccf)
        Volume in which values correspond to segmented viral projection density.

    normalized_projection_density : array-like, shape=(x_ccf, y_ccf, z_ccf)
        Volume in which values correspond to segmented viral projection density
        normalized by the total segmented injection volume
        (sum of injection density).

    centroid : array-like, shape=(1, 3)
        Spatial location of the injection centroid.

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
    INJECTION_HEMISPHERE = 2


    @staticmethod
    def _pull_data_volumes(mcc, experiment_id):
        """Pulls data volumes using mcc module.

        ...
         mcc.get_<data_volume> returns a tuple ( data_volume, meta_data )

        Parameters
        ----------
        """
        return {
            "data_mask" : mcc.get_data_mask(experiment_id)[0],
            "injection_density" : mcc.get_injection_density(experiment_id)[0],
            "injection_fraction" : mcc.get_injection_fraction(experiment_id)[0],
            "projection_density" : mcc.get_projection_density(experiment_id)[0]
        }

    @staticmethod
    def _mask_data_volume(arr, data_mask, tolerance=0):
        """Masks data in place
        """
        arr[ data_mask < tolerance ] = 0.

    @staticmethod
    def _compute_true_injection_density(injection_density, injection_fraction):
        """Computes 'true' injecton_density.

        Takes into consideration injection fracion...

        Parameters
        ----------
        """
        # inplace
        np.multiply(injection_density, injection_fraction, injection_density)

    @classmethod
    def from_mcc(cls, mcc, experiment_id):
        # pull data
        data_volumes = _pull_data_volumes( mcc, experiment_id )

        # compute 'true' injection density (inplace)
        _compute_true_injection_density( data_volumes["injection_density"],
                                         data_volumes["injection_fraction"] )

        # binarize mask
        data_mask = _get_bool_data_mask( data_volumes["data_mask"],
                                         cls.DATA_MASK_TOLERANCE )

        # mask data in place
        masker = partial( _mask_data_volume, data_mask=data_mask,
                          tolerance=cls.DATA_MASK_TOLERANCE )
        _ = map( masker, (data_volumes["injection_density"],
                          data_volumes["projection_density"]) )

        # return class
        return cls(injection_density=data_volumes["injection_density"],
                   projection_density=data_volumes["projection_density"])

    @staticmethod
    def _get_injection_hemisphere(injection_density):
        """Gets injection hemisphere based on injection density."""

        # split along depth dimension (forces arr.shape[2] % 2 == 0)
        l_hemi, r_hemi = np.split(injection_density, 2, axis=2)

        # return injecton hemisphere based on sum of injection_density
        if l_hemi.sum() > r_hemi.sum():
            return 1
        else:
            return 2

    @staticmethod
    def _flip_hemisphere(arr):
        """flips along 2 axis (hemipshere)

        ...
        Parameters
        ----------
        """
        return np.flip( arr, axis=2 )

    def __init__(self, injection_density=None, projection_density=None):

        if ( type(injection_density) == np.ndarray and
                type(projection_density) == np.ndarray ):

            if injection_density.shape != projection_density.shape:
                raise ValueError( "injection_density and projection_density "
                                  "must be the same shape!" )

        else:
            raise ValueError( "Both injection_density and projection_density "
                              "must be of type numpy.ndarray" )

        # check injection hemisphere
        computed_hemisphere = _get_injection_hemisphere( injection_density )

        if computed_hemisphere != self.INJECTION_HEMISPHERE:
            # flip experiment
            injection_density, projection_density = map(
                _flip_hemisphere, ( injection_density, projection_density )
            )

        self.injection_density = injection_density
        self.projection_density = projection_density

    @staticmethod
    def _compute_centroid(injection_density):
        """Computes centroid in index coordinates"""
        nonzero = injection_density[ injection_density.nonzero() ]
        voxels = np.argwhere( injection_density )

        return np.dot(nonzero, voxels) / injection_density.sum()

    @property
    def centroid(self):
        return compute_centroid(self.injection_density)

    @property
    def normalized_injection_density(self):
        return self.injection_density / self.injection_density.sum()

    @property
    def normalized_projection_density(self):
        return self.projection_density / self.injection_density.sum()

    def injection_ratio_contained(self, source_mask):
        """Returns the raito contained in a given mask.

        ...

        Parameters
        ----------
        """
        masked_injection = source_mask.mask_volume( self.injection_density )

        return masked_injection / injection_density.sum()
