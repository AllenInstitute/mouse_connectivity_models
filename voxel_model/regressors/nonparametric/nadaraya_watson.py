"""
Nadaraya-Watson Regression (also known as kernel regression)
"""

# Authors: Joseph Knox josephk@alleninstitute.org
# License:

# TODO : docs and example
# TODO : eval overwrite of K (kernel)
from __future__ import division

import numpy as np
from scipy.sparse import issparse

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.scorer import check_scoring
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection \
    import GridSearchCV, ParameterGrid, check_cv, _check_param_grid
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils import check_X_y


class NadarayaWatson(BaseEstimator, RegressorMixin):
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

    @staticmethod
    def _smoother(K, overwrite=False):
        """Computes nw weights"""
        factor = K.sum(axis=1)

        # we only want to normalize nonzero rows
        factor[factor == 0] = 1

        # divide in place
        if overwrite:
            return np.divide(K, factor[:, np.newaxis], K)

        return K/factor[:, np.newaxis]

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

    def get_weights(self, X):
        """Return model weights."""
        check_is_fitted(self, ["X_", "y_"])
        K = self._get_kernel(X, self.X_)

        return self._smoother(K, overwrite=True)

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

    @property
    def nodes(self):
        """Nodes (data)"""
        check_is_fitted(self, ["X_", "y_"])
        return self.y_


class _NadarayaWatsonLOOCV(NadarayaWatson):
    """Nadaraya watson with built-in Cross-Validation

    It allows efficient Leave-One-Out cross validatoin

    This class is not intended to be used directly. Use NadarayaWatsonCV instead.
    """
    def __init__(self, param_grid, scoring=None, store_cv_scores=False):
        self.param_grid = param_grid
        self.scoring = scoring
        self.store_cv_values = store_cv_scores
        _check_param_grid(param_grid)

    def _get_param_iterator(self):
        return ParameterGrid(self.param_grid)

    def _errors_and_values_helper(self, K):
        """Helper funciton to avoid duplication between self._errors and
        self._values.

        fill digonal with 0, renormalize
        """
        np.fill_diagonal(K, 0)
        S = self._smoother(K, overwrite=True) # also norms rows to 1

        return S

    def _errors(self, K, y):
        """ mean((y - Sy)**2) = mean( ((I-S)y)**2 )"""
        S = self._errors_and_values_helper(K)

        # I - S (S has 0 on diagonal)
        S *= -1
        np.fill_diagonal(S, 1.0)

        return np.mean((S.dot(y))**2)#linalg.norm(S.dot(y))

    def _values(self, K, y):
        """ prediction """
        S = self._errors_and_values_helper(K)

        return S.dot(y)

    def fit(self, X, y, sample_weight=None):
        """Fit Nadaraya Watson model


        X - arr of data centroids (#samples, #dim)
        y - arr of data projection (#samples, #voxels)
        """
        X, y = self._check_fit_arrays(X, y, sample_weight)

        candidate_params = list(self._get_param_iterator())

        scorer = check_scoring(self, scoring=self.scoring, allow_none=True)
        # error = scorer is None
        error = self.scoring is None

        if not error:
            # scorer wants an object to make predictions
            # but are already computed efficiently by _NadarayaWatsonCV.
            # This identity_estimator will just return them
            def identity_estimator():
                pass
            identity_estimator.predict = lambda y_pred: y_pred

        cv_scores = []
        for candidate in candidate_params:
            # NOTE: a bit hacky, find better way
            K = NadarayaWatson(**candidate)._get_kernel(X)
            if error:
                # NOTE: score not error!
                score = -self._errors(K, y)
            else:
                y_pred = self._values(K, y)
                score = scorer(identity_estimator, y, y_pred)
            cv_scores.append(score)

        self.n_splits_ = X.shape[0]
        self.best_index_ = np.argmax(cv_scores)
        self.best_score_ = cv_scores[self.best_index_]
        self.best_params_ = candidate_params[self.best_index_]
        if self.store_cv_scores:
            self.cv_scores_ = cv_scores

        return self


class NadarayaWatsonCV(NadarayaWatson):
    """NadarayaWatson Estimator with builtin loocv

    ...

    Parameters
    ----------
    """
    def __init__(self, param_grid, scoring=None, cv=None, store_cv_scores=False):
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.store_cv_scores = store_cv_scores

    def _update_params(self, param_dict):
        for k, v in param_dict.items():
            setattr(self, k, v)

    def fit(self, X, y, sample_weight=None):
        """Fit Nadaraya Watson estimator."""
        if self.cv is None:
            estimator = _NadarayaWatsonLOOCV(param_grid=self.param_grid,
                                             scoring=self.scoring,
                                             store_cv_scores=self.store_cv_scores)
            estimator.fit(X, y, sample_weight=sample_weight)
            self.best_score_ = estimator.best_score_
            self.n_splits_ = estimator.n_splits_
            best_params_ = estimator.best_params_
            if self.store_cv_values:
                self.best_index_ = estimator.best_index_
                self.cv_values_ = estimator.cv_values_
        else:
            if self.store_cv_values:
                raise ValueError("cv!=None and store_cv_score=True "
                                 "are incompatible")
            gs = GridSearchCV(NadarayaWatson(), self.param_grid,
                              cv=self.cv, scoring=self.scoring, refit=True)
            gs.fit(X, y, sample_weight=sample_weight)
            estimator = gs.best_estimator_
            self.best_score_ = gs.best_score_
            best_params_ = gs.best_params_

        # set params for predict
        self._update_params(best_params_)

        # store data for predict
        self.X_ = X
        self.y_ = y

        return self
