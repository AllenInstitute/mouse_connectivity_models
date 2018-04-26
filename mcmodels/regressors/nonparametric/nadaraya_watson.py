"""
Nadaraya-Watson Regression
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

# TODO : docs and example
# TODO : eval overwrite of K (kernel)
from __future__ import division

import numpy as np
from scipy.sparse import issparse

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.scorer import check_scoring
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import GridSearchCV, ParameterGrid, check_cv
from sklearn.model_selection._search import _check_param_grid
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

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _check_fit_arrays(self, X, y, sample_weight=None):
        """Checks fit arrays and scales y if sample_weight is not None."""
        # Convert data
        X, y = check_X_y(X, y, accept_sparse=("csr", "csc"),
                         multi_output=True, y_numeric=True)

        if sample_weight is not None and not isinstance(sample_weight, float):
            # TODO: break up?
            sample_weight = check_array(sample_weight, ensure_2d=False)

            # do not want to rescale X!!!!
            y = np.multiply(sample_weight[:, np.newaxis], y)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        return X, y

    def fit(self, X, y, sample_weight=None):
        """Fit Nadaraya Watson estimator.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.

        y : array, shape (n_samples, n_features)
            Target values.

        Returns
        -------
        self : returns an instance of self
        """
        X, y = self._check_fit_arrays(X, y, sample_weight)

        self.X_ = X
        self.y_ = y

        return self

    @staticmethod
    def _normalize_kernel(K, overwrite=False):
        """Normalizes kernel to have row sum == 1 if sum != 0"""
        factor = K.sum(axis=1)

        # if kernel has finite support, do not divide by zero
        factor[factor == 0] = 1

        # divide in place
        if overwrite:
            return np.divide(K, factor[:, np.newaxis], K)

        return K/factor[:, np.newaxis]


    def get_weights(self, X):
        """Return model weights."""
        check_is_fitted(self, ["X_", "y_"])
        K = self._get_kernel(X, self.X_)

        return self._normalize_kernel(K, overwrite=True)

    def predict(self, X):
        """Predict using the Nadaraya Watson model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        C : array, shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        check_is_fitted(self, ["X_", "y_"])

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        w = self.get_weights(X)

        # TODO: evaluate sklearn.utils.extmath.safe_sparse_dot()
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
        #TODO: check _check_param_grid in proper spot
        self.param_grid = param_grid
        self.scoring = scoring
        self.store_cv_scores = store_cv_scores
        _check_param_grid(param_grid)

    @property
    def _param_iterator(self):
        return ParameterGrid(self.param_grid)

    def _errors_and_values_helper(self, K):
        """Helper funciton to avoid duplication between self._errors and
        self._values.

        fill digonal with 0, renormalize
        """
        np.fill_diagonal(K, 0)
        S = self._normalize_kernel(K, overwrite=True)

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
        """Fit the model using efficient leave-one-out cross validation"""
        X, y = self._check_fit_arrays(X, y, sample_weight)

        candidate_params = list(self._param_iterator)

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
    """NadarayaWatson Estimator with built in Leave-one-out cross validation.

    By default, it performs Leave-one-out cross validation efficiently, but
    can accept cv argument to perform arbitrary cross validation splits.

    Parameters
    ----------
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values or a list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are
        explored. This enables searching over any sequence of parameter settings.

    scoring : string, callable or None, optional, default: None
        A string (see sklearn.model_evaluation documentation) or a scorer
        callable object / function with signature
        ``scorer(estimator, X, y)``

    cv : int, cross-validation generator or an iterable, optional, default: None
        Determines the cross-validation splitting strategy. If None, perform
        efficient leave-one-out cross validation, else use
        sklearn.model_selection.GridSearchCV.

    store_cv_scores : boolean, optional, default=False
        Flag indicating if the cross-validation values should be stored in
        `cv_scores_` attribute. This flag is only compatible with `cv=None`.

    Attributes
    ----------
    cv_scores_ : array, shape = (n_samples, ~len(param_grid))
        Cross-validation scores for each candidate parameter (if
        `store_cv_scores=True` and `cv=None`)

    best_score_ : float
        Mean cross-validated score of the best performing estimator.

    n_splits_ : int
        Number of cross-validation splits (folds/iterations)
    """
    def __init__(self, param_grid, scoring=None, cv=None, store_cv_scores=False,
                 kernel="linear", degree=3, coef0=1, gamma=None, kernel_params=None):
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.store_cv_scores = store_cv_scores

        # NadarayaWatson kwargs :: for compatibility
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

        self.kernel=kernel

    def _update_params(self, param_dict):
        for k, v in param_dict.items():
            setattr(self, k, v)

    def fit(self, X, y, sample_weight=None):
        """Fit Nadaraya Watson estimator.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.

        y : array, shape (n_samples, n_features)
            Target values.

        Returns
        -------
        self : returns an instance of self
        """
        if self.cv is None:
            estimator = _NadarayaWatsonLOOCV(param_grid=self.param_grid,
                                             scoring=self.scoring,
                                             store_cv_scores=self.store_cv_scores)
            estimator.fit(X, y, sample_weight=sample_weight)
            self.best_score_ = estimator.best_score_
            self.n_splits_ = estimator.n_splits_
            best_params_ = estimator.best_params_
            if self.store_cv_scores:
                self.best_index_ = estimator.best_index_
                self.cv_scores_ = estimator.cv_scores_
        else:
            if self.store_cv_scores:
                raise ValueError("cv!=None and store_cv_score=True "
                                 "are incompatible")
            gs = GridSearchCV(NadarayaWatson(), self.param_grid,
                              cv=self.cv, scoring=self.scoring, refit=True)
            gs.fit(X, y, sample_weight=sample_weight)
            estimator = gs.best_estimator_
            self.n_splits_ = gs.n_splits_
            self.best_score_ = gs.best_score_
            best_params_ = gs.best_params_

        # set params for predict
        self._update_params(best_params_)

        # store data for predict
        self.X_ = X
        self.y_ = y

        return self
