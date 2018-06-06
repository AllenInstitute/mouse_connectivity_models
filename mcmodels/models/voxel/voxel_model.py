"""
Module containing object :class:`VoxelModel`. The only difference between this
estimator and :class:`NadarayaWatson` is that class:`VoxelModel` one treats the
full injection volume (masked and flattened) as the input data `X` instead of
only the centroids. Thus, :class:'VoxelModel` must be instatiated with the
array of source_voxels, and the input data `X` must be passed as a tuple of arrays
: (centroids, injections) or a concatenated array.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

from __future__ import division
import numpy as np

from scipy.sparse import issparse
from sklearn.utils.validation import check_is_fitted

from ...regressors import NadarayaWatson


class VoxelModel(NadarayaWatson):
    """Voxel-scale interpolation model for mesoscale connectivity.

    Parameters
    ----------
    source_voxels : array-like, shape=(n_voxels, 3)
        List of voxel coordinates at which to interpolate.

    Examples
    --------
    >>> from mcmodels.core import VoxelModelCache
    >>> from mcmodels.models import VoxelModel
    >>> cache = VoxelModelCache
    >>> # get cortical experiment data
    >>> cortex_data = cache.get_experiment_data(injection_structure_ids=[315])
    >>> source_voxels = cortex_data.source_mask.coordinates
    >>> reg = VoxelModel(source_voxels)
    >>> reg.fit((cortex_data.centroids, cortex_data.injections))
    VoxelModel(source_voxels=array([[ ... ]]))

    References
    ----------
    Knox et al. 'High resolution data-driven model of the mouse connectome'.
        bioRxiv 293019; doi: https://doi.org/10.1101/293019

    See also
    --------
    NadarayaWatson
    """

    def __init__(self, source_voxels, kernel="linear", degree=3, coef0=1,
                 gamma=None, kernel_params=None):
        super(VoxelModel, self).__init__(kernel=kernel, degree=degree, coef0=coef0,
                                         gamma=gamma, kernel_params=kernel_params)

        self.source_voxels = source_voxels
        self.dimension = self.source_voxels.shape[1]

    def _compute_weights(self, X, y=None):
        K = self._get_kernel(X, y)
        return self._normalize_kernel(K, overwrite=True)

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
        if isinstance(X, (list, tuple)):
            centroids = X[0]
        else:
            # assume array
            centroids = X[:, :self.dimension]

        X, y = self._check_fit_arrays(centroids, y, sample_weight)

        self.y_ = y
        self.weights_ = self._compute_weights(self.source_voxels, centroids)

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
        C : array, shape=(X.shape[0], y.shape[1])
            Predicted normalized projection densities.

        """
        check_is_fitted(self, ["weights_", "y_"])

        if isinstance(X, (list, tuple)):
            injection = X[1]
        else:
            #assume array
            injection = X[:, self.dimension:]

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
        """Return model weights."""
        check_is_fitted(self, ["weights_", "y_"])
        return self.weights_

    @property
    def nodes(self):
        """Return model nodes (data)."""
        check_is_fitted(self, ["weights_", "y_"])
        return self.y_
