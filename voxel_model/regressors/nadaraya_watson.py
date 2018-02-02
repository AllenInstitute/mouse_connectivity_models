"""

"""

# Authors: Joseph Knox josephk@alleninstitute.org
# License:

# TODO : docs and example
# TODO : eval overwrite of K (kernel)

from __future__ import division
import numpy as np

from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted


class NadarayaWatson(BaseEstimator):
    """NadarayaWatson Estimator.

    see sklearn.kernel_ridge.KernelRidge for more info on parameters

    Parameters
    ----------
    kernel : string or callable, default="linear"
        Kernel mapping used to compute weights.

        see the documentation for sklearn.kernel_ridge.KernelRidge.

    gamma : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Ignored by other kernels.

        see the documentation for sklearn.metrics.pairwise.

    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of string to any, optional
        Additional parameters for kernel function passed as callable object.

    References
    ----------

    See also
    --------
    sklearn.kernel_ridge:
        Kernel Ridge Regression estimator from which the structure of
        this estimator is based.

    Examples
    --------
    >>> import numpy as np
    >>> from voxel_model.regressors import NadarayaWatson
    >>>

    """
    def __init__(self, kernel="linear", degree=3,
                 coef0=1, gamma=None, kernel_params=None):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def _get_kernel(self, X, y=None):
        """Gets kernel matrix."""
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}

        return pairwise_kernels(X, y, metric=self.kernel,
                                filter_params=True, **params)

    def _compute_weights(self, X, y=None):
        """Computes model weights."""
        K = self._get_kernel(X, y)
        factor = K.sum(axis=1)

        # we only want to normalize nonzero rows
        factor[factor == 0] = 1

        # divide in place
        np.divide(K, factor[:, np.newaxis], K)

        return K

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _check_fit_arrays(self, X, y, sample_weight):
        """Checks fit arrays and scales y if sample_weight is not None."""
        # Convert data
        X, y = check_X_y(X, y, accept_sparse=("csr", "csc"),
                         multi_output=True, y_numeric=True)

        if sample_weight is not None and not isinstance(sample_weight, float):
            sample_weight = check_array(sample_weight, ensure_2d=False)

            # dont want to rescale X!!!!
            y = np.multiply(sample_weight[:, np.newaxis], y)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        return X, y

    def fit(self, X, y, sample_weight=None):
        """
        """
        X, y = self._check_fit_arrays(X, y, sample_weight)

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """
        """
        check_is_fitted(self, ["X_", "y_"])

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        w = self.get_weights(X)

        if issparse(self.y_):
            # has to be of form sparse.dot(dense)
            # more efficient than w.dot( y_.toarray() )
            return self.y_.T.dot(w.T).T

        return w.dot(self.y_)

    def get_weights(self, X):
        """Return model weights."""
        check_is_fitted(self, ["X_", "y_"])
        return self._compute_weights(X, self.X_)

    @property
    def nodes(self):
        """Nodes (data)"""
        check_is_fitted(self, ["X_", "y_"])
        return self.y_
