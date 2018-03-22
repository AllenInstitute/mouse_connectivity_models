"""
Nonnegative Ridge Regression
"""

# Authors: Joseph Knox josephk@alleninstitute.org
# License:

# TODO : docs and example
import numpy as np

from sklearn.linear_model import _rescale_data
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils import check_consistent_length

from .base import NonnegativeLinear, _solve_nnls


def _solve_ridge_nnls(X, y, alpha):
    r"""Solves ...

    very similar to sklearn.linear_model.ridge._solve_lsqr

    $$ min ||Ax -y||_2^2 + \alpha\|x\|_2^2 $$
    $$ min x^T (A^T A + \alpha I) x + (-2 A^T y)^T x
    ...
    """
    # we set up as
    sqrt_alpha = np.sqrt(alpha)

    # append ridging matrix and zeros
    Q = X.T.dot(X) + np.diag(sqrt_alpha)
    c = -2*X.T.dot(y)

    # solve nnls system
    coef, res = _solve_nnls(Q, c)

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


class NonnegativeRidge(NonnegativeLinear):

    def __init__(self, alpha=1.0, normalize=False, copy_X=True):
        super(NonnegativeRidge, self).__init__(normalize=normalize,
                                               copy_X=copy_X)
        self.alpha = alpha

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
