"""
Nonnegative Ridge Regression.

:note: This is an experimental module.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

import numpy as np
from scipy import linalg

from sklearn.linear_model.base import _rescale_data
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils import check_consistent_length

from .base import NonnegativeLinear, _solve_nnls


def _solve_ridge_nnls(X, y, alpha):
    if X.ndim != 2 or y.ndim != 2:
        raise ValueError("X and y must be 2d arrays! May have to reshape "
                         "X.reshape(-1, 1) or y.reshape(-1, 1).")

    if alpha.size != X.shape[1]:
        raise ValueError("Number of targets and number of penalties "
                         "do not correspond: %d != %d"
                         % (alpha.size, X.shape[1]))

    # we set up as alpha**2
    sqrt_alpha = np.sqrt(alpha)

    # compute R^T R is more numerically stable than X^T X
    # 'r' mode returns tuple: (R,)
    R = linalg.qr(X, overwrite_a=False, mode='r', check_finite=False)[0]

    # rewrite as ||Ax - b||_2
    Q = R.T.dot(R) + np.diag(sqrt_alpha)
    c = X.T.dot(y)

    # solve nnls system
    coef, res = _solve_nnls(Q, c)

    return coef, res


def nonnegative_ridge_regression(X, y, alpha, sample_weight=None):
    r"""Solve the nonnegative least squares estimate regression problem.

    Solves

    .. math::
        \underset{x}{\text{argmin}} \| Ax - y \|_2^2 + \alpha^2 \| x \|_2^2
        \quad \text{for} \quad x \geq 0

    using `scipy.optimize.nnls <https://docs.scipy.org/doc/scipy/reference/
    generated/scipy.optimize.nnls.html>`_. This can be simplified to:

    .. math::

        \underset{x}{\text{argmin}} \| Qx - c \|_2^2 \quad \text{for} \quad x \geq 0

    where

    .. math::

        Q = X^TX + \alpha I \quad \text{and} \quad c = X^Ty

    Parameters
    ----------
    X : array, shape = (n_samples, n_features)
        Training data.

    y : array, shape = (n_samples,) or (n_samples, n_targets)
        Target values.

    alpha : float or array with shape = (n_features,)
        Regularization strength; must be a positive float. Improves the
        conditioning of the problem and reduces the variance of the estimates.
        Larger values specify stronger regularization.

    Returns
    -------
    coef : array, shape = (n_features,) or (n_features, n_targets)
        Weight vector(s).

    res : float
        The residual, :math:`\| Qx - c \|_2`

    Notes
    -----
    This is an experimental function.
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
    """Nonnegative least squares with L2 regularization.

    This model solves a regression model where the loss function is
    the nonnegative linear least squares function and regularization is
    given by the l2-norm. This estimator has built-in support for
    mulitvariate regression.

    Parameters
    ----------
    alpha : float or array with shape = (n_features,)
        Regularization strength; must be a positive float. Improves the
        conditioning of the problem and reduces the variance of the estimates.
        Larger values specify stronger regularization.

    Attributes
    ----------
    coef_ : array, shape = (n_features,) or (n_features, n_targets)
        Weight vector(s).

    res_ : float
        The residual, of the nonnegative least squares fitting.

    Examples
    --------
    >>> import numpy as np
    >>> from mcmodels.regressors import NonnegativeRidge
    >>> # generate some fake data
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y = np.random.randn(n_samples)
    >>> X = np.random.randn(n_samples, n_features)
    >>> # fit regressor
    >>> reg = NonnegativeRidge(alpha=1.0)
    >>> reg.fit(X, y)
    NonnegativeRidge(alpha=1.0)

    Notes
    -----
    This is an experimental class.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, sample_weight=None):
        """Fit nonnegative least squares linear model with L2 regularization.

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
        self.coef_, self.res_ = nonnegative_ridge_regression(
            X, y, alpha=self.alpha, sample_weight=sample_weight)

        return self
