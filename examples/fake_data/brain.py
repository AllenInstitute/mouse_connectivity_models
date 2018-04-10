"""
1D brain test case
"""
import numpy as np

import weight_profiles
from .base import _Experiment, BrainData

class Brain(object):
    """
    base brain class
    """

    def __init__(self, size=1000, profile='ridge_and_bump', profile_kwargs=None):
        self.size = size
        self.profile = profile
        self.profile_kwargs = profile_kwargs

        try:
            self.func = getattr(weight_profiles, self.profile)
        except AttributeError:
            raise ValueError("profile %s is not defined in weight_profiles"
                             % self.profile)

    def _compute_weights(self):
        return self.func(self.size, **self.profile_kwargs)

    @property
    def weights(self):
        try:
            return self._weights
        except AttributeError:
            self._weights = self._compute_weights
            return self._weights

    @property
    def domain(self):
        return np.arange(self.size)

    def get_experiments(self, n_samples, injection_fraction=0.1, noise_level=0,
                        normalize_injection=False, normalize_projection=False):
        """ yeild experiment arrays """
        centroids, injections, projections = [], [], []
        for _ in n_samples:
            experiment = _Experiment.from_brain(
                self, injection_fraction, noise_level)

            if normalize_injection:
                experiment.normalize_injection()

            if normalize_projection:
                experiment.normalize_projection()

            centroids.append(experiment.centroid)
            injections.append(experiment.injection)
            projections.append(experiment.projection)

        return BrainData(centroids, injections, projections)
