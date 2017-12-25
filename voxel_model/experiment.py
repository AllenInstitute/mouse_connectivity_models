"""

"""
from __future__ import division, relative_import
import numpy as np

from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi

from .masks import SourceMask, TargetMask

class Experiment(object):
    """Experiment class
    """

    def __init__(self, mcc, experiment_id):
        self.mcc = mcc
        self.experiment_id = experiment_id

    @property
    def data_mask(self):
        return self.mcc.get_data_mask(self.experiment_id)[0]

    def _mask_to_valid(self, data):
        """Masks data to data mask"""
        data[self.data_mask.nonzero()] = 0.0
        return data

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
