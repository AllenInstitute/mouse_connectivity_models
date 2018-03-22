# Authors: Joseph Knox josephk@alleninstitute.org
# License:
# TODO : docs and example
from __future__ import print_function, absolute_import, division

import numpy as np
import scipy.optimize as sopt


from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearModel, _rescale_data
from sklearn.metrics import mean_squared_error
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import check_cv, LeaveOneOut
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils import check_consistent_length



#class _BaseNNRidge(six.with_metaclass(ABCMeta, BaseEstimator)):
#
#    @abstractmethod
#    def __init__(self, alpha=1.0):
#        self.alpha = alpha
#
#    @staticmethod
#    def _fit_w(C, d):
#        """returns weights"""
#        w = np.empty((C.shape[1], d.shape[1]))
#        for j, col in enumerate(d.T):
#            w[:, j] = sopt.nnls(C, col)[0]
#
#        return w
#
#    @abstractmethod
#    def fit(self, X, y, sample_weight=None):
#        """fit data from model"""
#
#    def predict(self, X):
#        """Predict unionized y
#        """
#        check_is_fitted(self, ["weights_"])
#        if len(X.shape) == 1:
#            X = X.reshape(-1, 1)
#
#        return X.dot(self.weights_)


def _solve_ridge_nnls(X, y, alpha):
    """Solves ...

    very similar to sklearn.linear_model.ridge._solve_lsqr

    $$ min ||Ax -y||_2^2 + \alpha\|x\|_2^2 $$
    $$ min x^T (A^T A + \alpha I) x + (-2 A^T y)^T x
    ...
    """
    n_samples, n_features = X.shape
    coef = np.empty((y.shape[1], n_features), dtype=X.dtype)
    res = np.empty(y.shape[1], dtype=np.float64)

    # we set up as
    sqrt_alpha = np.sqrt(alpha)

    # append ridging matrix and zeros
    #Q = np.vstack((X, np.diag(sqrt_alpha)))
    #c = np.vstack((y, np.zeros((n_features, y.shape[1]))))
    Q = X.T.dot(X) + np.diag(sqrt_alpha)
    c = -2*X.T.dot(y)

    for i in range(c.shape[1]):
        c_column = c[:, i]
        info = sopt.nnls(Q, c_column)

        coef[i] = info[0]
        res[i] = info[1]

    return coef, res


def nonnegative_ridge_regression(X, y, alpha, sample_weight=None):
    """Nonnegative ridge regression.

    Very similar to sklearn.linear_model.ridge.ridge_regression, but uses
    nonnegativity constraint.
    ...
    """
    # TODO accept_sparse=['csr', 'csc', 'coo']? check sopt.nnls
    # TODO order='F'?
    X = check_array(X)
    y = check_array(y, ensure_2d=False)
    check_consistent_length(X, y)

    n_samples, n_features = X.shape

    ravel = False
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        ravel = True

    n_samples_, n_targets = y.shape

    if n_samples != n_samples_:
        raise ValueError("Number of samples in X and y does not correspond:"
                         " %d != %d" % (n_samples, n_samples_))

    has_sw = sample_weight is not None

    if has_sw:
        if np.atleast_1d(sample_weight).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        X, y = _rescale_data(X, y, sample_weight)

    # there should be either 1 or n_targets penalties
    # NOTE: different from sklearn.linear_model.ridge
    alpha = np.asarray(alpha, dtype=X.dtype).ravel()
    if alpha.size not in [1, n_features]:
        raise ValueError("Number of targets and number of penalties "
                         "do not correspond: %d != %d"
                         % (alpha.size, n_features))

    # NOTE: different from sklearn.linear_model.ridge
    if alpha.size == 1 and n_features > 1:
        alpha = np.repeat(alpha, n_features)

    coef, res = _solve_ridge_nnls(X, y, alpha)

    if ravel:
        # When y was passed as 1d-array, we flatten the coefficients
        coef = coef.ravel()

    return coef, res


class NonnegativeRidge(LinearModel, RegressorMixin):

    # we do not allow fitting of intercept for now
    fit_intercept = False
    intercept = 0

    def __init__(self, alpha=1.0, normalize=False, copy_X=True):
        self.alpha = alpha
        self.normalize = normalize
        self.copy_X = copy_X

    def fit(self, X, y, sample_weight=None):
        """ Fit Oh

        X - regional, unionized
        y - regional, unionized
        """
        # TODO: add support for sparse
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        if X.ndim == 1:
            # NOTE may be unnecessary/bad???
            X = X.reshape(-1, 1)

        if ((sample_weight is not None) and
                np.atleast_1d(sample_weight).ndim > 1):
            raise ValueError("Sample weights must be 1D array or scalar")

        # NOTE: self.fit_intercept=False
        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X,
            sample_weight=sample_weight)

        # fit weights
        self.coef_, self.res_ = nonnegative_ridge_regression(
            X, y, alpha=self.alpha, sample_weight=sample_weight)

        return self

    @property
    def weights(self):
        """Convenience property for pulling out regional matrix."""
        check_is_fitted(self, ["coef_"])
        if 
        return self.coef_.T


class _NonnegativeRidgeGCV(NonnegativeRidge):
    """Linear Model with built in cv
    Not to be used directly
    """
    def __init__(self, alphas=(1e-1, 1.0, 1e1), cv=None, scoring=None, store_cv_values=False):
        self.alphas = alphas
        self.cv = cv
        self.scoring = scoring
        self.store_cv_values = store_cv_values

        if self.cv is None:
            self.cv = LeaveOneOut()

    def fit(self, X, y, sample_weight=None):
        """ Fit Linear Model

        X - regional, unionized
        y - regional, unionized
        """
        # TODO: add support for sparse
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        if ((sample_weight is not None) and
                np.atleast_1d(sample_weight).ndim > 1):
            raise ValueError("Sample weights must be 1D array or scalar")
        n_samples, n_features = X.shape

        # NOTE: self.fit_intercept=False
        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X,
            sample_weight=sample_weight)

        if sample_weight is not None:
            X, y = _rescale_data(X, y, sample_weight)

        n_y = 1 if y.ndims == 1 else y.shape[1]
        cv_values = np.empty((n_samples * n_y, len(self.alphas)))
        C = []

        scorer = check_scoring(self, scoring=self.scoring, allow_none=True)
        # error = scorer is None
        error = self.scoring is None

        if not error:
            # scorer wants an object to make predictions
            # but are already computed efficiently by _OhModelCV.
            # This identity_estimator will just return them
            def identity_estimator():
                pass
            identity_estimator.predict = lambda y_pred: y_pred

        for i, alpha in enumerate(self.alphas):
            for i, (train, test) in enumerate(self.cv.split(X, y)):
                w = super(_NonnegativeRidgeGCV, self)._fit_w(X[train], y[train])
                y_pred = X[test].dot(w)

                if error:
                    # NOTE: score not error!
                    score = -mean_squared_error(y_pred, y[test], sample_weight=sample_weight)
                else:
                    score = scorer(identity_estimator, y, y_pred)

                cv_values[i] = score
                W.append(w)

        best_idx = np.argmax(cv_values)
        self.best_score_ = cv_values[best_idx]
        self.weights_ = W[best_idx]

        if self.store_cv_values:
            self.cv_values_ = cv_values

        return self


class NonnegativeRidgeCV(NonnegativeRidge):
    """Oh model with built in cv.

    A utility class for easy nested cross validation.
    """
    def __init__(self, alphas=(1e-3, 1.0, 1e3), scoring=None, cv=None, store_cv_values=False):
        self.alphas = alphas
        self.scoring = scoring
        self.cv = cv
        self.store_cv_values = store_cv_values

    def fit(self, X, y, sample_weight=None):
        """Fit Oh estimator."""
        estimator = _NonnegativeRidgeGCV(scoring=self.scoring, cv=self.cv,
                                         store_cv_values=self.store_cv_values)
        estimator.fit(X, y, sample_weight=sample_weight)
        self.best_score_ = estimator.best_score_
        self.weights_ = estimator.weights_
        if self.store_cv_values:
            self.cv_values_ = estimator.cv_values_

        return self
