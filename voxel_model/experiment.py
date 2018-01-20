# Authors: Joseph Knox josephk@alleninstitute.org
# License:
from __future__ import division
import numpy as np

class _Experiment(object):
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

    _DATA_KEYS = {
        "injection_density" : "get_injection_density",
        "injection_fraction" : "get_injection_fraction",
        "projection_density" : "get_projection_density"
    }

    def __init__(self, mcc, experiment_id):
        self.mcc = mcc
        self.experiment_id = experiment_id

        # pull data
        self.data_mask = self.mcc.get_data_mask(self.experiment_id)[0]

        for data_key, data_function in self._DATA_KEYS.items():
            # pull data and mask to valid
            data = getattr(self.mcc, data_function)(self.experiment_id)[0]
            setattr(self, data_key, self._mask_to_valid(data))

        self.centroid = self._get_centroid()

        # flip if wrong hemi
        if self.centroid[2] < self.data_mask.shape[2]//2:
            self._flip_data()

    def _flip_data(self):
        """flip all data"""
        self.data_mask = self.data_mask[...,::-1]

        for data_key in self._DATA_KEYS.keys():
            # flip each volume
            data = getattr(self, data_key)
            setattr(self, data_key, data[...,::-1])

        self.centroid = self._get_centroid()


    def _get_centroid(self):
        # get centroid
        return self.mcc.api.calculate_injection_centroid(self.injection_density,
                                                         self.injection_fraction,
                                                         resolution=1)

    def _mask_to_valid(self, data):
        """Masks data to data mask

        data_mask is not binary! It represents the fraction of the voxel
        that is 'valid' data. We choose 0.5 as a threshold

        Parameters
        ----------
        data : array-like, shape=
        """
        data[self.data_mask < 0.5] = 0.0
        return data

    @property
    def normalized_projection_density(self):
        return np.divide(self.projection_density, self.injection_density.sum())

    @property
    def normalized_injection_density(self):
        return np.divide(self.injection_density, self.injection_density.sum())

class ModelData(object):
    """Container for model data...

    ...
    ...
    ...

    """
    def _get_experiments(self):
        """  ... """
        # masked columns
        xcols = self.source_mask.nonzero
        ycols = self.target_mask.nonzero

        X, y, centroids, total_volumes = [], [], [], []
        for experiment_id in self.experiment_ids:
            # get experiment data
            exp = _Experiment(self.mcc, experiment_id)

            # for min_ratio_contained
            total_volumes.append( exp.normalized_injection_density.sum() )

            # update
            X.append( exp.normalized_injection_density[xcols] )
            y.append( exp.normalized_projection_density[ycols] )
            centroids.append( exp.centroid )

        # stack centroids, injections
        X = np.hstack( (np.asarray(centroids), np.asarray(X)) )

        # return arrays
        return X, np.asarray(y), np.asarray(total_volumes)

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

        # get valid rows
        self.valid_rows = self._get_valid_rows()

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
