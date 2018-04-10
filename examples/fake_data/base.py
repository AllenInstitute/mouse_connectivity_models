import random
from collections import namedtuple

import numpy as np

class _Experiment(object):

    @classmethod
    def from_brain(cls, brain, injection_fraction, noise_level):
        def get_center_and_bounds(radius):
            center = random.random()
            lower = max(0, center - radius)
            upper = min(1, center + radius)

            return center, (lower, upper)

        def get_idx(bounds):
            return slice(*map(lambda x: int(brain.size * x), bounds))

        def get_injection(idx):
            injection = np.zeros(brain.size)
            injection[idx] = 1.

            return injection

        def get_projection(idx, noise_level):
            projection = brain.weights[idx].sum(axis=1)
            return projection + np.random.normal(0, noise_level, projection.size)

        radius = 0.5 * injection_fraction
        center, bounds = get_center_and_bounds(radius)

        idx = get_idx(bounds)

        centroid = brain.size * center
        injection = get_injection(idx)
        projection = get_projection(idx, noise_level)

        return cls(centroid, injection, projection)

    def __init__(self, centroid, injection, projection):
        if injection.shape != projection.shape:
            raise ValueError("injection and projection must have same shape")

        self.centroid = centroid
        self.injection = injection
        self.projection = projection

        self.injection_sum = injection.sum()

        self.injection_normalized = False
        self.projection_normalized = False

    def normalize_injection(self):
        if not self.injection_normalized:
            self.injection /= self.injection_sum
            self.injection_normalized = True

            return self

    def normalize_projection(self):
        if not self.projection_normalized:
            self.projection /= self.injection_sum
            self.projection_normalized = True

            return self

class BrainData(namedtuple('BrainData',
                           ('centroids', 'injections', 'projections'))):

    __slots__ = ()

    def __new__(cls, centroids, injections, projections):
        centroids = np.asarray(centroids)
        injections = np.asarray(injections)
        projections = np.asarray(projections)

        if injections.shape != projections.shape:
            raise ValueError("injection and projection must have same shape")

        if centroids.shape[0] != injections.shape[0]:
            raise ValueError("injections and centroids must have the same "
                             "number of samples")

        return cls(centroids, injections, projections)
