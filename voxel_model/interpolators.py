# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import division
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels, check_pairwise_arrays
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

class VoxelModel(BaseEstimator):
    """Voxel scale interpolation model for mesoscale connectivity.

    Model details can be found at <PAPER>.

    Functions similar/identical to sklearn.kernel_ridge.KernelRidge:
        * _get_kernel
        * _pairwise
    Also the documentation for the following is taken verbatim:
        * kernel
        * gamma
        * degree
        * coef0
        * kernel_params

    Parameters
    ----------
    source_voxels : array-like, shape=(n_voxels, 3)
        List of voxel coordinates at which to interpolate.

    epsilon : float, optional (default=0)
        Nonnegative regularization parameter, similar to a ridging parameter.

    kernel : string or callable, default="linear"
        Kernel mapping used internally. A callable should accept two arguments
        and the keyword arguments passed to this object as kernel_params, and
        should return a floating point number.

    gamma : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Interpretation of the default value is left to
        the kernel; see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.

    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.

    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : mapping of string to any, optional
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.

    Attributes
    ----------
    centroids_fit_ : array, shape=(n_exps, 3)
        Centroid coordinates of the injections used to fit the model.

    y_fit_ : array, shape=(n_exps, n_target_voxels)
        The projection volume in the target for each experiment.

    weights_ : array, shape=(n_source_voxels, n_exps)
        The fitted weights matrix.

    References
    ----------
    * Joseph Knox ...
      "High Resolution Voxel ...."

    See also
    --------
    sklearn.kernel_ridge:
        Kernel Ridge Regression estimator from which this estimator is based.

    Examples
    --------
    >>> from voxel_model.interpolators import VoxelModel
    >>> import numpy as np
    >>> n_exps = 20
    >>> n_source_voxels, n_target_voxels = 125, 200
    >>> source_voxels = np.argwhere( np.ones((5,5,5))) )
    >>> injections = np.random.randn(n_exps, n_source_voxels)
    >>> centroids = source_voxels[ np.random.choice(n_source_voxels,
    >>>                                             n_exps,
    >>>                                             replacement=False) ]
    >>> X = np.hstack((centroids, injections))
    >>> y = np.random.randn(n_exps, n_target_voxels)
    >>> reg = VoxelModel(source_voxels, kernel="rbf", gamma=1.5)
    >>> reg.fit(X, y)
    >>>
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
        """taken from sklearn.kernel_ridge"""
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}

        return pairwise_kernels(X, y, metric=self.kernel,
                                filter_params=True, **params)

    def _get_weights(self, centroids):
        K = self._get_kernel(self.source_voxels, centroids)

        # normalize by row sum
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
        """Fit Voxel Model.

        NOTE : X is a concatenation (column wise) of the injection centroid
            coordinates and the injection volumes. This choice was made to
            be consistent with the sklearn.core.BaseEstimator fit and predict
            schema

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=(n_exps, 3+n_source_voxels)
            Centroid coordinates concatenated with the injection density for
            each training experiment.

        y : {array-like, sparse matrix}, shape=(n_exps, n_target_voxels)
            Normalized projection density for each training experiment

        Returns
        -------
        self : returns an instance of self.
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
        """Predict projection volumes given injection volumes.

        NOTE : X is a concatenation (column wise) of the injection centroid
            coordinates and the injection volumes. This choice was made to
            be consistent with the sklearn.core.BaseEstimator fit and predict
            schema

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=(n_exps, 3+n_source_voxels)
            Centroid coordinates concatenated with the injection density for
            each test experiment.

        Returns
        -------
        C : array, shape=(X.shape[0], y_fit_.shape[1])
            Predicted normalized projection densities.
        """
        check_is_fitted(self, ["weights_"])

        # if len(X.shape) == 1:
        #     X = X.reshape(-1, 1)

        X_predict_ = X[:,self.dimension:]

        return X_predict_.dot(self.weights_).dot(self.y_fit_)

    def get_voxel_matrix(self):
        """Produces the full n_source_voxels x n_target_voxels connectivity.

        NOTE : when dealing with real data, this function will likely produce
            an array on the order of TBs and will most likely not fit in memory.
            Only use on small subsets of real data or on toy data.

        Returns
        -------
        C : array, shape=(n_source_voxels, n_target_voxels)
            The full voxel x voxel connectivity.
        """
        check_is_fitted(self, ["weights_"])

        return self.weights_.dot(self.y_fit_)

class RegionalizedVoxelModel(object):
    """Regionalization/Parcelation of VoxelModel.

    Regionalizes the connectivity model in VoxelModel given a brain parcelation.

    Parameters
    ----------
    see VoxelModel for:
        * source_voxels
        * epsilon
        * kernel
        * degree
        * coef0
        * gamma
        * kernel_params

    voxel_model : VoxelModel object, optional (default=None)
        A VoxelModel object. The default instatiates a VoxelModel estimator
        using the default setttings.

        See VoxelModel for more details.

    source_key : array-like, shape=(n_source_voxels,)
        Flattened key relating each source voxel to a given brain region.

    target_key : array-like, shape=(n_target_voxels,)
        Flattened key relating each target voxel to a given brain region.

    Examples
    --------
    >>> from voxel_model.interpolators import RegionalizedVoxelModel
    >>> import numpy as np
    >>> n_exps = 20
    >>> n_source_voxels, n_target_voxels = 125, 200
    >>> source_voxels = np.argwhere( np.ones((5,5,5))) )
    >>> injections = np.random.randn(n_exps, n_source_voxels)
    >>> centroids = source_voxels[ np.random.choice(n_source_voxels,
    >>>                                             n_exps,
    >>>                                             replacement=False) ]
    >>> X = np.hstack((centroids, injections))
    >>> y = np.random.randn(n_exps, n_target_voxels)
    >>> source_key = np.random.randint(0,10, size=n_source_voxels)
    >>> target_key = np.random.randint(0,10, size=n_target_voxels)
    >>> reg = RegionalizedVoxelModel(source_voxels, source_key, target_key)
    >>> reg.fit(X, y)
    >>> reg.get_region_matrx().shape
    (10,10)
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
        """Fits the underlying VoxelModel estimator.

        NOTE : X is a concatenation (column wise) of the injection centroid
            coordinates and the injection volumes. This choice was made to
            be consistent with the sklearn.core.BaseEstimator fit and predict
            schema

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=(n_exps, 3+n_source_voxels)
            Centroid coordinates concatenated with the injection density for
            each training experiment.

        y : {array-like, sparse matrix}, shape=(n_exps, n_target_voxels)
            Normalized projection density for each training experiment

        Returns
        -------
        self : returns an instance of self.
        """
        self.voxel_model.fit(X,y)

    def predict(self, X, normalize=False):
        """Predict regionalized connectivity given injection volumes.

        NOTE : X is a concatenation (column wise) of the injection centroid
            coordinates and the injection volumes. This choice was made to
            be consistent with the sklearn.core.BaseEstimator fit and predict
            schema

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=(n_exps, 3+n_source_voxels)
            Centroid coordinates concatenated with the injection density for
            each test experiment.

        Returns
        -------
        C : array, shape=(n_source_regions, n_target_regions)
            Predicted regionalized connectivity.
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
        """Produces the full regionalized connectivity

        Parameters
        ----------
        metric : string, optional (default="connection_strength")
            Metric with which to represent the regionalized connectivity.
            Valid choices are:
                * "connection_strength" (default)
                    W = w_ij |X||Y|
                    The sum of the voxel-scale connectivity between each pair
                    of source-target regions.

                * "connection_density"
                    W = w_ij |X|
                    The average voxel-scale connectivity between each source
                    voxel to each source region.

                * "normalized_connection_strength"
                    W = w_ij |Y|
                    The average voxel-scale connectivity between each source
                    region to each target voxel"

                * "normalized_connection_density"
                    W = w_ij
                    The average voxel-scale connectivity between each pair of
                    source-target regions

        Returns
        -------
        C : array-like, shape=(n_source_regions, n_target_regions)
            The regionalized voxel-scale connectivity.
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
