# Authors: Joseph Knox josephk@alleninstitute.org
# License:
# TODO : docs and example
from __future__ import division
from abc import ABCMeta, abstractmethod

import numpy as np
import numpy.linalg as LA
import scipy.special as sp
import scipy.optimize as sopt

from scipy.sparse import issparse
from scipy.spatial.distance import pdist, cdist, squareform

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import check_cv, LeaveOneOut
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six


class _BaseNNRidge(six.with_metaclass(ABCMeta, BaseEstimator)):

    @abstractmethod
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    @staticmethod
    def _fit_w(C, d):
        """returns weights"""
        w = np.empty((C.shape[1], d.shape[1]))
        for j, col in enumerate(d.T):
            w[:, j] = sopt.nnls(C, col)[0]

        return w

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """fit data from model"""

    def predict(self, X):
        """Predict unionized y
        """
        check_is_fitted(self, ["weights_"])
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        return X.dot(self.weights_)

def nonnegative_ridge_regression(X, y, alpha, sample_weight=None):
    """Nonnegative ridge regression.

    ...
    """
    if alpha != 0.0:
        # append
        X_ = np.vstack((X, alpha*np.eye(X.shape[1])))
        y_ = np.vstack((y, np.zeros((X.shape[1], y.shape[1]))))
    else:
        X_ = X
        y_ = y

    w = np.empty((X_.shape[1], y_.shape[1]))
    for j, col in enumerate(y_.T):
        w[:, j] = sopt.nnls(X_, col)[0]

    return w.T


class NNRidge(LinearModel, RegressorMixin):

    def __init__(self, alpha=1.0, normalize=False, copy_X=True):
        self.alpha = alpha
        self.normalize = normalize
        self.copy_X = copy_X

    def fit(self, X, y, sample_weight=None):
        """ Fit Oh

        X - regional, unionized
        y - regional, unionized
        """
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # MAY BE BAD
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)


        # fit weights
        self.coef_ = nonnegative_ridge_regression(X, y, alpha=self.alpha,
                                                  sample_weight=sample_weight)

        return self

    @property
    def weights(self):
        """Convenience property for pulling out regional matrix."""
        check_is_fitted(self, ["coef_"])
        return self.coef_.T


class _LinearModelGCV(LinearModel):
    """Linear Model with built in cv
    Not to be used directly
    """
    def __init__(self, alphas=(1e-1, 1, 1e1), cv=None, scoring=None, store_cv_values=False):
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
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # MAY BE BAD
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        W = []
        cv_values = np.empty(len(X))
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

        for i, (train, test) in enumerate(self.cv.split(X, y)):
            w = super(_LinearModelCV, self)._fit_w(X[train], y[train])
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


class LinearModelCV(LinearModel):
    """Oh model with built in cv.

    A utility class for easy nested cross validation.
    """
    def __init__(self, alpha=1000, scoring=None, cv=None, store_cv_values=False):
        super(LinearModelCV, self).__init__(alpha=alpha)
        self.scoring = scoring
        self.cv = cv
        self.store_cv_values = store_cv_values

    def fit(self, X, y, sample_weight=None):
        """Fit Oh estimator."""
        estimator = _LinearModelCV(scoring=self.scoring, cv=self.cv,
                                   store_cv_values=self.store_cv_values)
        estimator.fit(X, y, sample_weight=sample_weight)
        self.best_score_ = estimator.best_score_
        self.weights_ = estimator.weights_
        if self.store_cv_values:
            self.cv_values_ = estimator.cv_values_

        return self
