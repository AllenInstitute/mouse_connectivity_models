# Authors: Joseph Knox josephk@alleninstitute.org
# License: 

from __future__ import division, absolute_import
import numpy as np

from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi

from .masks import SourceMask, TargetMask

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
    data_mask : array-like, shape (x_ccf, y_ccf, z_ccf)
        Mask of invalid voxels. 
    injection_density : array-like, shape (c_ccf, y_ccf, z_ccf)
        Volume in which values correspond to segmented viral injection density.
    injection_fraction : array-like, shape (c_ccf, y_ccf, z_ccf)
        Volume in which values correspond to segmented viral injection fraction.
        In other words, the fraction of the voxel that lies within the annotation.
    projection_density : array-like, shape (c_ccf, y_ccf, z_ccf)
        Volume in which values correspond to segmented viral projection density.
    normalized_projection_density : array-like, shape (c_ccf, y_ccf, z_ccf)
        Volume in which values correspond to segmented viral projection density
        normalized by the total segmented injection volume 
        (sum of injection density).
    centroid : array-like, shape (1, 3)
        Spatial location of the injection centroid.

    Examples
    --------

    >>> from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    >>> from voxel_model.experiment import Experiment
    >>> mcc = MouseConnectivityCache(resolution=100)
    >>> eid = 126862385 
    >>> exp = Experiment(mcc, eid)
    >>> exp.injection_density.shape
    (132,80,114)
    """

    def __init__(self, mcc, experiment_id):
        self.mcc = mcc
        self.experiment_id = experiment_id

    @property
    def data_mask(self):
        return self.mcc.get_data_mask(self.experiment_id)[0]

    @property
    def injection_density(self):
        ind = self.mcc.get_injection_density(self.experiment_id)[0]
        return self._mask_to_valid(ind)

    @property
    def injection_fraction(self):
        inf = self.mcc.get_injection_fraction(self.experiment_id)[0]
        return self._mask_to_valid(inf)

    @property
    def projection_density(self):
        prd = self.mcc.get_projection_density(self.experiment_id)[0]
        return self._mask_to_valid(prd)

    @property
    def normalized_projection_density(self):
        return self.projection_density/self.injection_density.sum()

    @property
    def centroid(self):
        return MouseConnectivityApi().calculate_injection_centroid(
            self.injection_fraction, self.injection_density, resolution=1
        )

    def _mask_to_valid(self, data):
        """Masks data to data mask"""
        data[self.data_mask.nonzero()] = 0.0
        return data

class ModelData(object): 
    """Container for data used in model
    """

    normalized = True

    def __init__(self, mcc, experiment_ids=None, structure_ids=None, 
                 source_mask=None, target_mask=None, hemisphere=3):

        self.mcc = mcc
        self.hemisphere = hemisphere
        
        if experiment_ids is not None and structure_ids is not None:
            self.experiment_ids = experiment_ids
            self.structure_ids = strucutre_ids
        else:
            return ValueError("currently both exp_ids and str_ids must be provided")

        # get source/target masks, ensure consistent
        self.source_mask = SourceMask(mcc, self.structure_ids)
        self.target_mask = TargetMask(mcc, self.structure_ids, self.hemisphere)

        # 
        self._get_experiment_data()

    def _get_experiment_data(self):
        """Returns the attrs from exp
        """
        centroids = []
        injections = []
        projections = []
        for eid in self.experiment_ids:
            # get experiment data
            experiment = Experiment(self.mcc, eid)

            # mask injection/projection to source/target masks
            ind = experiment.injection_density[self.source_mask.where]
            prd = experiment.normalized_projection_density[self.target_mask.where]

            # append relevant attrs
            centroids.append(experiment.centroid)
            injections.append(ind)
            projections.append(prd)

        # make np arrays
        self.injections = np.asarray(injecitons)
        self.projections = np.asarray(projections)
        self.centroids = np.asarray(centroids)
