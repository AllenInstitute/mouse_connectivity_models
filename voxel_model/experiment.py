# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import division
from collections import namedtuple
import numpy as np

class Experiment(namedtuple("Experiment", ["injection_density",
                                           "projection_density"])):
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
    __slots__ = ()

    DATA_MASK_TOLERANCE = 0.5

    @classmethod
    def from_mcc(cls, mcc, experiment_id, injection_hemisphere=2):
        # pull data
        data_mask = mcc.get_data_mask(experiment_id)[0]
        injection_density = mcc.get_injection_density(experiment_id)[0]
        injection_fraction = mcc.get_injection_fraction(experiment_id)[0]
        projection_density = mcc.get_projection_density(experiment_id)[0]

        # compute 'true' injection density (inplace)
        np.multiply(injection_density, injection_fraction, injection_density)

        # mask
        projection_density[ data_mask < cls.DATA_MASK_TOLERANCE ] = 0.
        injection_density[ data_mask < cls.DATA_MASK_TOLERANCE ] = 0.

        return cls(injection_density=injection_density,
                   projection_density=projection_density,
                   injection_hemisphere=injection_hemisphere)

    @staticmethod
    def _get_injection_hemisphere(injection_density):
        """Gets injection hemisphere based on injection density."""

        # split along depth dimension (forces arr.shape[2] % 2 == 0)
        l_hemi, r_hemi = np.split(injection_density, 2, axis=2)

        if l_hemi.sum() > r_hemi.sum():
            return 1
        else:
            return 2

    def __new__(cls, injection_density=None, projection_density=None,
                injection_hemisphere=2):
        if ( type(injection_density) == np.ndarray and
                type(projection_density) == np.ndarray ):

            if injection_density.shape != projection_density.shape:
                raise ValueError( "injection_density and projection_density "
                                  "must be the same shape!" )

        else:
            raise ValueError( "Both injection_density and projection_density "
                              "must be of type numpy.ndarray" )


        if injection_hemisphere in [1,2]:
            # check injection hemisphere
            computed_hemi = cls._get_injection_hemisphere(injection_density)

            if injection_hemisphere != computed_hemi:
                # flip experiment
                injection_density = np.flip( injection_density, axis=2 )
                projection_density = np.flip( projection_density, axis=2 )

        elif injection_hemisphere != 3:
            raise ValueError( "injection_hemisphere must be 1, 2, or 3" )

        return super(Experiment, cls).__new__( cls, injection_density,
                                               projection_density )

    @staticmethod
    def compute_centroid(injection_density):
        """Computes centroid in index coordinates"""
        nonzero = injection_density[ injection_density.nonzero() ]
        voxels = np.argwhere( injection_density )

        return np.dot(nonzero, voxels) / injection_density.sum()

    @property
    def centroid(self):
        return self.compute_centroid(self.injection_density)

    @property
    def normalized_injection_density(self):
        return self.injection_density / self.injection_density.sum()

    @property
    def normalized_projection_density(self):
        return self.projection_density / self.injection_density.sum()
