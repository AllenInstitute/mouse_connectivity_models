"""
Nonnegative Linear Regression
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

# TODO : docs and example
import numpy as np
import scipy.optimize as sopt

from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel, _rescale_data
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils import check_consistent_length


def _solve_nnls(X, y):
    if X.ndim != 2 or y.ndim != 2:
        raise ValueError("X and y must be 2d arrays! May have to reshape "
                         "X.reshape(-1, 1) or y.reshape(-1, 1).")

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
    """Solve the nonnegative least squares estimate regression problem.

    Solves ``argmin_x \| Ax - y \|_2^2`` for ``x > 0`` using scipy.optimize.nnls

    Parameters
    ----------
    X : array, shape = (n_samples, n_features)
        Training data.

    y : array, shape = (n_samples,) or (n_samples, n_targets)
        Target values.

    Returns
    -------
    coef : array, shape = (n_features,) or (n_samples, n_features)
        Weight vector(s).

    res : float
        The residual, ``\| Ax - y \|_2``
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
    """Nonnegative least squares linear model.

    This model solves a regression model where the loss function is the
    nonnegative linear least squares function. This estimator has built-in
    support for mulitvariate regression.

    Attributes
    ----------
    coef_ : array, shape = (n_features,) or (n_features, n_targets)
        Weight vector(s).

    res_ : float
        The residual, of the nonnegative least squares fitting.

    Examples
    --------
    >>> from voxel_model.regressors import NonnegativeLinear
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y = np.random.randn(n_samples)
    >>> X = np.random.randn(n_samples, n_features)
    >>> reg = NonnegativeLinear()
    >>> reg.fit(X, y)
    NonnegativeLinear()
    """

    # needed for compatibility with LinearModel.predict() (decision_function)
    intercept_ = 0

    def fit(self, X, y, sample_weight=None):
        """Fit nonnegative least squares linear model.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            Training data.

        y : array, shape = (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        # TODO: add support for sparse
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if ((sample_weight is not None) and
                np.atleast_1d(sample_weight).ndim > 1):
            raise ValueError("Sample weights must be 1D array or scalar")

        # fit weights
        self.coef_, self.res_ = nonnegative_regression(
            X, y, sample_weight=sample_weight)

        return self
