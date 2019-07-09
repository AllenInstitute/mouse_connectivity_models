from __future__ import division
from math import log10, ceil, sqrt
import itertools
import logging
import abc
import six

import numpy as np
from scipy.spatial.distance import cdist
from mcmodels.regressors.nonparametric.kernels import Polynomial


class _BaseParams(six.with_metaclass(abc.ABCMeta)):

    @abc.abstractproperty
    def param_grid(self):
        """ """

class RBFParams(_BaseParams):

    DEFAULT_SIGMAS = np.round(np.logspace(0.3, 1.4, 10), 2)

    def __init__(self, sigmas=None):
        if sigmas is None:
            sigmas = self.DEFAULT_SIGMAS
        self.sigmas = sigmas

    @property
    def gammas(self):
        return 1 / (self.sigmas**2)

    @property
    def param_grid(self):
        return dict(kernel=['rbf'], gamma=self.gammas)


class PolynomialParams(_BaseParams):

    DEFAULT_SHAPES = np.round(np.logspace(0, 2.7, 12))

    def __init__(self, voxel_data, shapes=None, search_support=False):
        if shapes is None:
            shapes = self.DEFAULT_SHAPES
        self.shapes = shapes
        self.voxel_data = voxel_data
        self.search_support = search_support

    @staticmethod
    def support_bounds(voxels, centroids):
        logging.debug("Computing max minimum distance from any centroid")
        D = cdist(voxels, centroids, metric='sqeuclidean')

        # ceiling of max(min(d)) and max(d)
        bound = lambda arr: ceil(sqrt(arr.max()))
        return bound(D.min(axis=1)), bound(D)

    @property
    def param_grid(self):
        source_voxels = self.voxel_data.injection_mask.coordinates
        a, b = self.support_bounds(source_voxels, self.voxel_data.centroids)
        logging.debug("support_bounds: (%.0f, %.0f)", a, b)

        if self.search_support:
            supports = np.round(np.linspace(a, 2*b, 10))
        else:
            supports = [a]

        return dict(kernel=[Polynomial(shape=a, support=b)
                            for a, b in itertools.product(self.shapes, supports)])
