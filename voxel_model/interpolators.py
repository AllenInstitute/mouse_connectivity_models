"""

"""
from __future__ import division
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels, check_pairwise_arrays
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted


class VoxelModel(BaseEstimator):
    """
    similar to sklearn.kernel_ridge.KernelRidge
    """
    def __init__(self, source_voxels, epsilon=0, kernel="linear", degree=3,
                 coef0=1, gamma=None, kernel_params=None):

        self.source_voxels = source_voxels
        self.dimension = self.source_voxels.shape[1]
        if epsilon >= 0:
            self.epsilon = epsilon
        else:
            raise ValueError("epsilon must be nonnegative")

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
        """ """

    def _get_weights(self, centroids):

        K = self._get_kernel(self.source_voxels, centroids)

        factor = K.sum(axis=1)
        if self.epsilon:
            factor += self.epsilon
        else:
            # esure no zero rows for division
            factor = np.where(factor != 0, factor, 1)

        # divide in place
        np.divide(K, factor[:,np.newaxis], K)

        return K

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def fit(self, X, y, sample_weight=None):
        """
        """
        # Convert data
        X, y = check_X_y(X, y, accept_sparse=("csr", "csc"),
                         multi_output=True, y_numeric=True)

        if sample_weight is not None and not isinstance(sample_weight, float):
            sample_weight = check_array(sample_weight, ensure_2d=False)

            # dont want to rescale X!!!!
            y = np.multiply(sample_weight[:,np.newaxis], y)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # centroids are dense, rest is sparse
        self.centroids_fit_ = X[:,:self.dimension]#.toarray()
        self.y_fit_ = y

        self.weights_ = self._get_weights(self.centroids_fit_)

        return self

    def predict(self, X):
        """
        """
        check_is_fitted(self, ["weights_"])

        # if len(X.shape) == 1:
        #     X = X.reshape(-1, 1)

        X_predict_ = X[:,self.dimension:]

        return X_predict_.dot(self.weights_).dot(self.y_fit_)

    def get_voxel_matrix(self):
        """
        """
        check_is_fitted(self, ["weights_"])

        return self.weights_.dot(self.y_fit_)


class RegionalizedVoxelModel(object):
    """
    """

    valid_regional_metrics = [
        "connection_strength",
        "connection_density",
        "normalized_connection_strength",
        "normalized_connection_density"
    ]

    def __init__(self, source_voxels, source_key, target_key, 
                 voxel_model=None, epsilon=0, kernel="linear", degree=3,
                 coef0=1, gamma=None, kernel_params=None):
        """ FIND WAY TO USE CLSMETHOD """
        if not voxel_model is None:
            self.voxel_model = voxel_model
        else:
            self.voxel_model = VoxelModel(source_voxels,
                                          epsilon=epsilon,
                                          kernel=kernel,
                                          degree=degree,
                                          coef0=coef0, 
                                          gamma=gamma, 
                                          kernel_params=kernel_params)

        self.source_key = source_key
        self.target_key = target_key

    def fit(self, X, y):
        """ """
        self.voxel_model.fit(X,y)

    def predict(self, X, normalize=False):
        """
        """
        # reshape 1xn
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # predict from grid level
        voxel_prediction = self.voxel_model.predict(X)

        # get target
        t_regions = np.unique(target_key)

        # return array
        n_pred = X.shape[0]
        nt_regions = len(t_regions)
        region_pred = np.empty((n_pred, nt_regions))

        for ii, region in enumerate(t_regions):
            cols = np.isin(target_key, region)
            # note, if region were 1 voxel, would not work
            region_pred[:,ii] = voxel_prediction[:,cols].sum(axis=1)

        if normalize:
            inj_vols = X[:,self.dimension:].sum(axis=1)
            np.divide(region_pred, inj_vols[:,np.newaxis], region_pred)

        return region_pred

    def get_region_matrix(self, metric="connection_strength"):
        """
        """
        if metric not in self.valid_regional_metrics:
            raise ValueError(
                "metric must be one of {self.valid_regional_metrics}"
            )

        # note, already fit, just used to return region weights
        t_regions, t_counts = np.unique(target_key, return_counts=True)
        s_regions, s_counts = np.unique(source_key, return_counts=True)

        ns_regions = len(s_regions)
        nt_regions = len(t_regions)
        ns_points = self.weights_.shape[0]

        # integrate target regions
        # NOTE: probably more efficient to sort then stride by nt_regions
        temp = np.empty([nt_regions, ns_points])
        for ii, region in enumerate(t_regions):
            cols = np.isin(target_key, region)
            temp[ii,:] = self.weights_.dot(
                np.einsum('ji->j', self.y_fit_[:,cols])
            )

        # integrate source regions
        # NOTE: probably more efficient to sort then stride by ns_regions
        region_matrix = np.empty([ns_regions, nt_regions])
        for ii, region in enumerate(s_regions):
            cols = np.isin(source_key, region)
            # note, if region were 1 voxel, would not work
            region_matrix[ii,:] = temp[:,cols].sum(axis=1)

        if metric == "connection_strength":
            # w_ij |X||Y|
            return region_matrix
        elif metric == "connection_density":
            # w_ij |X|
            return np.divide(region_matrix,
                             t_counts[np.newaxis,:], 
                             region_matrix)
        elif metric == "normalized_connection_strength":
            # w_ij |Y|
            return np.divide(region_matrix,
                             s_counts[:,np.newaxis],
                             region_matrix)
        else:
            # normalized_connection_density
            # w_ij
            return np.divide(region_matrix, 
                             np.outer(s_counts, t_counts),
                             region_matrix)
 
# class NadarayaWatson(BaseEstimator):
#     """
#     95% from sklearn.kernel_ridge.KernelRidge
#     """
#     def __init__(self, kernel="linear", degree=3,
#                  coef0=1, gamma=None, kernel_params=None):
#         self.kernel = kernel
#         self.gamma = gamma
#         self.degree = degree
#         self.coef0 = coef0
#         self.kernel_params = kernel_params
# 
#     def _get_kernel(self, X, y=None):
#         if callable(self.kernel):
#             params = self.kernel_params or {}
#         else:
#             params = {"gamma": self.gamma,
#                       "degree": self.degree,
#                       "coef0": self.coef0}
# 
#         return pairwise_kernels(X, y, metric=self.kernel,
#                              filter_params=True, **params)
# 
#     def _get_weights(self, X, y=None):
# 
#         K = self._get_kernel(X, y)
# 
#         factor = K.sum(axis=1)
#         factor = np.where(factor > 0, factor, 1)
# 
#         # divide in place
#         np.divide(K, factor[:,None], K)
#         return K
# 
#     @property
#     def _pairwise(self):
#         return self.kernel == "precomputed"
# 
#     def fit(self, X, y):
#         """
#         """
#         # Convert data
#         X, y = check_X_y(X, y, accept_sparse=("csr", "csc"),
#                          multi_output=True, y_numeric=True)
# 
#         if len(y.shape) == 1:
#             y = y.reshape(-1, 1)
# 
#         self.X_fit_ = X
#         self.y_fit_ = y
# 
#         return self
# 
#     def predict(self, X):
#         """
#         """
#         check_is_fitted(self, ["X_fit_", "y_fit_"])
#         w = self._get_weights(X, self.X_fit_)
#         return np.dot(w, self.y_fit_)
# 
#     def voxel_weights(self, X):
#         return self._get_weights(X, self.X_fit_)
