# Authors: Joseph Knox josephk@alleninstitute.org
# License:

# TODO : eval overwrite of K (kernel)

from __future__ import division
import numpy as np

from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted


class NadarayaWatson(BaseEstimator):
    """
    95% from sklearn.kernel_ridge.KernelRidge
    """
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

        # we only want to normalize nonzero rows
        factor[ factor==0 ] = 1

        # divide in place
        np.divide(K, factor[:,np.newaxis], K)

        return K

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _check_fit_arrays(self, X, y, sample_weight):
        # Convert data
        X, y = check_X_y(X, y, accept_sparse=("csr", "csc"),
                         multi_output=True, y_numeric=True)

        if sample_weight is not None and not isinstance(sample_weight, float):
            sample_weight = check_array(sample_weight, ensure_2d=False)

            # dont want to rescale X!!!!
            y = np.multiply(sample_weight[:,np.newaxis], y)

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
        else:
            return w.dot( self.y_ )

    def get_weights(self, X):
        check_is_fitted(self, ["X_", "y_"])
        return self._compute_weights(X, self.X_)

    @property
    def nodes(self):
        check_is_fitted(self, ["X_", "y_"])
        return self.y_
