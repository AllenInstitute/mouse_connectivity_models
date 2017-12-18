"""

"""
from __future__ import division
import numpy as np

from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi

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
