# Authors: Joseph Knox josephk@alleninstitute.org
# License:

# NOTE : REMOVED epsilon parameter

from __future__ import division, absolute_import
import numpy as np

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

    def _stack_X(self, X):
        """Helper function if tuple is passed as X = (centroids, X)."""

        help_string = "\nIs your tuple :: X = (centroids, injections)?\n"

        if len(X) != 2:
            raise ValueError( "tuple must be length 2." + help_string )

        if X[0].shape[1] != self.dimension:
            raise ValueError( "centroids array (X[0]) has the wrong "
                              "dimension." + help_string )

        if X[1].shape[1] != self.source_voxels.shape[0]:
            raise ValueError( "injection array (X[1]) has wrong number of "
                              "voxels (columns)." + help_string )

        # stack centroids and injections horizontally (column wise)
        return np.hstack(X)

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
        if isinstance(X, tuple):
            X = self._stack_X(X)

        X, y = self._fit(X, y, sample_weight)

        # centroids are dense, rest is sparse
        self.y_ = y

        # cleave off centroids
        centroids = X[:,:self.dimension]
        self.weights_ = self._compute_weights( centroids )

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
        if isinstance(X, tuple):
            X = self._stack_X(X)

        check_is_fitted(self, ["weights_", "y_"])

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        injection = X[:,self.dimension:]

        return injection.dot(self.weights_).dot(self.y_)

    def get_weights(self):
        check_is_fitted(self, ["weights_", "y_"])
        return self.weights_

    @property
    def weights(self):
        check_is_fitted(self, ["weights_", "y_"])
        return self.weights_
