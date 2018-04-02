"""
Polynomial Kernel
"""

# Authors: Joseph Knox josephk@alleninstitute.org
# License:

# TODO : docs and example
# TODO: eval_gradient
from __future__ import division

import numpy as np
import scipy.special as sp
from scipy.spatial.distance import pdist, cdist, squareform

from sklearn.gaussian_process.kernels \
    import Kernel, StationaryKernelMixin, NormalizedKernelMixin, Hyperparameter


class _BasePolynomial(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Base for polynomial kerel ...
    """
    def __init__(self, shape=1.0, support=1.0, support_bounds=(0, 1e5)):
        if shape < 0:
            raise ValueError("shape must be nonnegative")

        self.shape = shape
        self.support = support
        self.support_bounds = support_bounds

    @property
    def anisotropic(self):
        return False

    @property
    def hyperparameter_support(self):
        return Hyperparameter("support", "numeric", self.support_bounds)

    @property
    def coefficient(self):
        """coefficient scaling the kernel to have int_D K(u)du == 1"""
        return 1.0 / (self.support * sp.beta(0.5, self.shape + 1))

    def __call__(self, X, Y=None, eval_gradient=False):
        """return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        """
        X = np.atleast_2d(X)

        # we may not want this
        shape = self.shape # _check_length_scale
        support = self.support

        if Y is None:
            dists = pdist(X, metric='sqeuclidean')

            # convert from upper-triangular matrix to square matrix
            # (diag has all 0.0)
            K = squareform(dists)
            np.multiply(K, support**-2, out=K)
        else:
            dists = cdist(X, Y, metric='sqeuclidean')
            K = np.multiply(dists, support**-2)

        # equiv to (1-(d/(length_scale**2))**decay
        K.clip(max=1.0, out=K)
        np.subtract(1, K, out=K)
        np.power(K, shape, out=K)

        if eval_gradient:
            #if self.hyperparameter_support.fixed and self.hyperparameter_shape.fixed:
            #    K_gradient = np.empty((X.shape[0], X.shape[0], 0))
            #else:
            #    K_gradient = np.multiply(K, squareform(dists))[:, :, np.newaxis]

            #return K, K_gradient
            raise NotImplementedError
        else:
            return K

    def __repr__(self):
        return "{0}(support={1:.3g}, shape={2:.3g})".format(
            self.__class__.__name__, np.ravel(self.support)[0], np.ravel(self.shape)[0])


class Polynomial(_BasePolynomial):
    """Polynomial function kernel.

    It is stationary ...


    k(x, x') =

    Parameters
    -----------

    """
    def __init__(self, shape=1.0, support=1.0, shape_bounds=(1e-5, 1e5),
                 support_bounds=(0, 1e5)):
        super(Polynomial, self).__init__(shape=shape,
                                         support=support,
                                         support_bounds=support_bounds)
        self.shape_bounds = shape_bounds

    @property
    def hyperparameter_shape(self):
        return Hyperparameter("shape", "numeric", self.shape_bounds)


class Uniform(_BasePolynomial):
    """Convenience for ..."""

    SHAPE = 0

    def __init__(self, support=1.0, support_bounds=(0, 1e5)):
        super(Uniform, self).__init__(shape=Uniform.SHAPE,
                                      support=support,
                                      support_bounds=support_bounds)


class Epanechnikov(_BasePolynomial):
    """Convenience for ..."""

    SHAPE = 1

    def __init__(self, support=1.0, support_bounds=(0, 1e5)):
        super(Epanechnikov, self).__init__(shape=Epanechnikov.SHAPE,
                                           support=support,
                                           support_bounds=support_bounds)


class Biweight(_BasePolynomial):
    """Convenience for ..."""

    SHAPE = 2

    def __init__(self, support=1.0, support_bounds=(0, 1e5)):
        super(Biweight, self).__init__(shape=Biweight.SHAPE,
                                       support=support,
                                       support_bounds=support_bounds)


class Triweight(_BasePolynomial):
    """Convenience for ..."""

    SHAPE = 3

    def __init__(self, support=1.0, support_bounds=(0, 1e5)):
        super(Triweight, self).__init__(shape=Triweight.SHAPE,
                                        support=support,
                                        support_bounds=support_bounds)
