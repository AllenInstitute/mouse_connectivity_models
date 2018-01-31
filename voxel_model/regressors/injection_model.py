# Authors: Joseph Knox josephk@alleninstitute.org
# License:

# NOTE : REMOVED epsilon parameter

from __future__ import division, absolute_import
import operator as op
import numpy as np

from scipy.sparse import issparse
from sklearn.utils.validation import check_is_fitted

from . import NadarayaWatson


class InjectionModel(NadarayaWatson):
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
    def __init__(self, source_voxels, **kwargs):

        super(InjectionModel, self).__init__(**kwargs)

        self.source_voxels = source_voxels
        self.dimension = self.source_voxels.shape[1]

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
        if isinstance(X, tuple) or isinstance(X, list):
            centroids = X[0]
        else:
            # assume array
            centroids = X[:,:self.dimension]

        X, y = self._check_fit_arrays(centroids, y, sample_weight)

        self.y_ = y
        self.weights_ = self._compute_weights( self.source_voxels, centroids )

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
        check_is_fitted(self, ["weights_", "y_"])

        if isinstance(X, tuple):
            injection = X[1]
        else:
            #assume array
            injection = X[:,self.dimension:]

        if len(injection.shape) == 1:
            injection = injection.reshape(-1, 1)

        # has to be of form sparse.dot(dense)
        y = self.y_.toarray() if issparse(self.y_) else self.y_

        return injection.dot(self.weights_).dot(y)

    def get_weights(self):
        """Overwrite of NadarayaWatson.get_weights."""
        check_is_fitted(self, ["weights_", "y_"])
        return self.weights_

    @property
    def weights(self):
        check_is_fitted(self, ["weights_", "y_"])
        return self.weights_
