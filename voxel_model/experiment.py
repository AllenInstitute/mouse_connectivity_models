# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import division
import numpy as np

class _ExperimentData(object):
    """ ...

    ...
    ...

    Parameters
    ----------
    """
    @classmethod
    def from_mcc(cls, mcc, experiment_id):
        # pull data
        injection_density = mcc.get_injection_density(experiment_id)[0]
        injection_fraction = mcc.get_injection_fraction(experiment_id)[0]
        projection_density = mcc.get_projection_density(experiment_id)[0]

        return cls(injection_density=injection_density,
                   injection_fraction=injection_fraction,
                   projection_density=projection_density)

    def __init__(self, injection_density=None, injection_fraction=None,
                 projection_density=None):
        self.injection_density = injection_density
        self.injection_fraction = injection_fraction
        self.projection_density = projection_density

    def __dict__(self):
        return ["injection_density", "injection_fraction", "projection_density"]

    @property
    def injection_volume(self):
        return self.injection_density.sum()

    @property
    def injection_hemisphere(self):
        """Defined by hemisphere with majoirty of injection."""
        midline = self.injection_density.shape[2]//2
        l_inj_vol = self.injection_density[:,:,:midline].sum()
        r_inj_vol = self.injection_density[:,:,midline:].sum()

        if l_inj_vol > r_inj_vol:
            # left hemisphere
            return 1
        else:
            # right hemisphere
            return 2

    def mask_to_valid(self, data_mask):
        """ ... """
        if not data_mask.dtype == bool:
            raise ValueError("data_mask must be boolean")

        for var in self.__dict__():
            arr = getattr(self, var)
            arr[ ~data_mask ] = 0.
            setattr(self, var, arr)

    def flip_hemisphere(self):
        """ ... """
        for var in self.__dict__():
            arr = getattr(self, var)
            arr = arr[..., ::-1]
            setattr(self, var, arr)


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

    def __init__(self, mcc, experiment_id):
        self.mcc = mcc
        self.experiment_id = experiment_id

        # get api
        self.api = self.mcc.api

        # get experiment data
        self.data = _ExperimentData.from_mcc(self.mcc, self.experiment_id)

        # mask data to valid
        data_mask = self.mcc.get_data_mask(self.experiment_id)[0]
        self.data.mask_to_valid( data_mask > self.DATA_MASK_TOLERANCE )

        # flip if wrong hemisphere
        if self.data.injection_hemisphere != self.INJECTION_HEMISPHERE:
            self.data.flip_hemisphere()


    @property
    def centroid(self):
        # get centroid
        return self.api.calculate_injection_centroid(self.data.injection_density,
                                                     self.data.injection_fraction,
                                                     resolution=1)

    @property
    def normalized_projection_density(self):
        return np.divide(self.data.projection_density, self.data.injection_volume)

    @property
    def normalized_injection_density(self):
        return np.divide(self.data.injection_density, self.data.injection_volume)
