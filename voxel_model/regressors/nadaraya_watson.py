# Authors: Joseph Knox josephk@alleninstitute.org
# License:

# TODO : eval overwrite of K (kernel)

from __future__ import division
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels, check_pairwise_arrays
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

class NadarayaWatson(BaseEstimator):
    """
    95% from sklearn.kernel_ridge.KernelRidge
    """
    def __init__(self, kernel="linear", degree=3,
                 coef0=1, gamma=None, kernel_params=None):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def _get_kernel(self, X, y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}

        return pairwise_kernels(X, y, metric=self.kernel,
                                filter_params=True, **params)

    def _compute_weights(self, X, y=None):

        K = self._get_kernel(X, y)

        factor = K.sum(axis=1)
        factor[ factor==0 ] = 1 #prevent division error

        # divide in place
        np.divide(K, factor[:,np.newaxis], K)

        return K

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def fit(self, X, y):
        """
        """
        # Convert data
        X, y = check_X_y(X, y, accept_sparse=("csr", "csc"),
                         multi_output=True, y_numeric=True)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

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
        return np.dot(w, self.y_)

    def get_weights(self, X):
        check_is_fitted(self, ["X_", "y_"])
        return self._compute_weights(X, self.X_)

    @property
    def nodes(self):
        check_is_fitted(self, ["X_", "y_"])
        return self.y_
