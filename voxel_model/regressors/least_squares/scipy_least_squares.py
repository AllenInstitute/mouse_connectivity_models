"""
Scikit-Learn wrapping for scipy.optimize.least_squares
"""

# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: BSD 3

from abc import ABCMeta, abstractmethod
import six

import numpy as np
import scipy.optimize as sopt

from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel, _rescale_data
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils import check_consistent_length


class ScipyLeastSquares(six.with_metaclass(ABCMeta, RegressorMixin)):
    """scikit-learn estimator wrapping scipy.optimize.least_squares"""

    def __init__(self, x0, **least_squares_kwargs):
        self.x0 = x0
        self.least_squares_kwargs = least_squares_kwargs

    def residuals(self, X, y, coeffs=None):
        """Computes residuals of function."""
        if coeffs is None:
            check_is_fitted(self, "self.opt_result_")
            coeffs = self.opt_result_.x

        return self._predict(coeffs, X) - y

    def fit(self, X, y, sample_weight=None, **least_squares_kwargs):
        """Fits with optional update of kwargs"""
        # TODO: add support for sparse
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        if X.ndim == 1:
            # NOTE may be unnecessary/bad???
            X = X.reshape(-1, 1)

        if sample_weight is not None:
            if np.atleast_1d(sample_weight).ndim > 1:
                raise ValueError("Sample weights must be 1D array or scalar")

            X, y = _rescale_data(X, y, sample_weight)


        # scipy.optimize.least_squares kwargs
        lstsq_kws = self.least_squares_kwargs
        lstsq_kws.update(least_squares_kwargs)

        # fit scipy.optimize.least_squares
        opt_result = sopt.least_squares(self.residuals,
                                        self.x0,
                                        args=(X, y),
                                        **lstsq_kws)

        # scipy.optimize.OptimizeResult (namedtuple)
        self.opt_result_ = opt_result

        return self

    @abstractmethod
    def _predict(self, coeffs, X):
        """general prediction given X"""

    def predict(self, X):
        """Predict given model fit"""
        check_is_fitted(self, "self.opt_result_")
        coeffs = self.opt_result_.x

        return self._predict(coeffs, X)
