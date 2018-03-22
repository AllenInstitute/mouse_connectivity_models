"""
Nonnegative Linear Regression
"""

# Authors: Joseph Knox josephk@alleninstitute.org
# License:

# TODO : docs and example
import numpy as np
import scipy.optimize as sopt

from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearModel, _rescale_data
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils import check_consistent_length


def _solve_nnls(X, y):
    """Solves ..

    """
    n_features = X.shape[1]
    n_targets = y.shape[1]
    coef = np.empty((n_targets, n_features), dtype=X.dtype)
    res = np.empty(n_targets, dtype=np.float64)

    for i in range(n_targets):
        y_column = y[:, i]
        info = sopt.nnls(X, y_column)

        coef[i] = info[0]
        res[i] = info[1]

    return coef, res


def nonnegative_regression(X, y, sample_weight=None):
    """Nonnegative regression.

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

    coef, res = _solve_nnls(X, y)

    if ravel:
        # When y was passed as 1d-array, we flatten the coefficients
        coef = coef.ravel()

    return coef, res


class NonnegativeLinear(LinearModel, RegressorMixin):

    # we do not allow fitting of intercept for now
    fit_intercept = False
    intercept = 0

    def __init__(self, normalize=False, copy_X=True):
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
        self.coef_, self.res_ = nonnegative_regression(
            X, y, sample_weight=sample_weight)

        return self

    @property
    def weights(self):
        """Convenience property for pulling out regional matrix."""
        check_is_fitted(self, ["coef_"])
        return self.coef_.T
