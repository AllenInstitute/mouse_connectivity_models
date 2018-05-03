"""
Nonnegative Elastic-Net Regression.

:note: This is an experimental module.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

import warnings

import numpy as np
import scipy.linalg as linalg
import scipy.optimize as sopt

from sklearn.linear_model.base import _rescale_data
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils import check_consistent_length

from .base import NonnegativeLinear, _solve_nnls


def _solve_elastic_net_nnls(A, b, alpha, rho, solver, **solver_kwargs):
    """Solves nonnegative elastic net through quadratic programming."""
    n_features = A.shape[1]
    n_targets = b.shape[1]

    # define loss and gradient functions
    loss = lambda x: x.T.dot(Q).dot(x) + c.dot(x)
    grad = lambda x: (Q.T + Q).dot(x) + c

    # compute R^T R is more numerically stable than X^T X
    # 'r' mode returns tuple: (R,)
    R = linalg.qr(A, overwrite_a=False, mode='r', check_finite=False)[0]

    Q = R.T.dot(R) + np.diag(alpha)
    C = rho[:, np.newaxis] - 2*A.T.dot(b)

    # sopt.minimize params
    x0 = np.ones(n_features)
    bounds = tuple(zip(n_features*[0.0], n_features*[None]))

    # return arrays
    coef = np.empty((n_targets, n_features), dtype=A.dtype)
    res = np.empty(n_targets, dtype=A.dtype)

    for i in range(n_targets):
        c = C[:, i]
        sol = sopt.minimize(loss, x0, jac=grad, method=solver, bounds=bounds,
                            **solver_kwargs)

        if not sol.success:
            warnings.warn('Optimization was not a success for column %d, '
                          'treat results accordingly' % i)

        coef[i] = sol.x
        res[i] = sol.fun + np.inner(c, c)

    return coef, res


def nonnegative_elastic_net_regression(X, y, alpha, rho, sample_weight=None,
                                       solver='L-BFGS-B', **solver_kwargs):
    if solver not in ('L-BFGS-B', 'TNC', 'SLSQP'):
        raise ValueError('solver must be one of L-BFGS-B, TNC, SLSQP, '
                         'not %s' % solver)

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
    alpha = np.asarray(alpha, dtype=X.dtype).ravel()
    if alpha.size not in [1, n_features]:
        raise ValueError("Number of targets and number of L2 penalties "
                         "do not correspond: %d != %d"
                         % (alpha.size, n_features))

    rho = np.asarray(rho, dtype=X.dtype).ravel()
    if rho.size not in [1, n_features]:
        raise ValueError("Number of targets and number of L1 penalties "
                         "do not correspond: %d != %d"
                         % (rho.size, n_features))


    # NOTE: different from sklearn.linear_model.ridge
    if alpha.size == 1 and n_features > 1:
        alpha = np.repeat(alpha, n_features)
    if rho.size == 1 and n_features > 1:
        rho = np.repeat(rho, n_features)

    coef, res = _solve_elastic_net_nnls(X, y, alpha, rho, solver, **solver_kwargs)

    if ravel:
        # When y was passed as 1d-array, we flatten the coefficients
        coef = coef.ravel()

    return coef, res


class NonnegativeElasticNet(NonnegativeLinear):
    """
    Notes
    -----
    This is an experimental class.
    """

    def __init__(self, alpha=1.0, rho=1.0, solver='L-BFGS-B', **solver_kwargs):
        if solver not in ('L-BFGS-B', 'TNC', 'SLSQP'):
            raise ValueError('solver must be one of L-BFGS-B, TNC, SLSQP, '
                             'not %s' % solver)
        self.alpha = alpha
        self.rho = rho
        self.solver = solver
        self.solver_kwargs = solver_kwargs

    def fit(self, X, y, sample_weight=None):
        """Fit nonnegative least squares linear model with elastic-net regularization.

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
        self.coef_, self.res_ = nonnegative_elastic_net_regression(
            X, y, alpha=self.alpha, sample_weight=sample_weight,
            solver=self.solver, **self.solver_kwargs)

        return self
