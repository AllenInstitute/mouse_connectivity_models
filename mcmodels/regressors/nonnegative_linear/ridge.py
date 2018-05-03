"""
Nonnegative Ridge Regression.

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

from .base import NonnegativeLinear


def _solve_ridge_nnls(A, b, alpha, solver, **solver_kwargs):
    """Solves nonnegative ridge regressiond through quadratic programming."""
    # compute R^T R is more numerically stable than A^T A
    # 'r' mode returns tuple: (R,)
    R = linalg.qr(A, overwrite_a=False, mode='r', check_finite=False)[0]

    # x^T Q x + C_col x
    Q = R.T.dot(R) + np.diag(alpha**2)
    C = -2*A.T.dot(b)

    # define loss and gradient functions
    loss = lambda x: x.T.dot(Q).dot(x) + c.dot(x)
    grad = lambda x: (Q.T + Q).dot(x) + c

    n_features = A.shape[1]
    n_targets = b.shape[1]

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
        res[i] = sol.fun + b[:, i].T.dot(b[:, i])

    return coef, res


def nonnegative_ridge_regression(X, y, alpha, sample_weight=None,
                                 solver='SLSQP', **solver_kwargs):
    r"""Solve the nonnegative least squares estimate ridge regression problem.

    Solves

    .. math::
        \underset{x}{\text{argmin}} \| Ax - b \|_2^2 + \alpha^2 \| x \|_2^2
        \quad \text{s.t.} \quad x \geq 0

    We can write this as the quadratic programming (QP) problem:

    .. math::

        \underset{x}{\text{argmin}} x^TQx - c^Tx \quad \text{s.t.} \quad x \geq 0

    where

    .. math::

        Q = A^TA + \alpha I \quad \text{and} \quad c = -2A^Ty

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

    sample_weight : float or array-like, shape (n_samples,), optional (default = None)
        Individual weights for each sample.

    solver : string, optional (default = 'SLSQP')
        Solver with which to solve the QP. Must be one that supports bounds
        (i.e. 'L-BFGS-B', 'TNC', 'SLSQP').

    **solver_kwargs
        See `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/
        reference/generated/scipy.optimize.minimize.html>`_
        for valid keyword arguments

    Returns
    -------
    coef : array, shape = (n_features,) or (n_features, n_targets)
        Weight vector(s).

    res : float
        The residual, :math:`\| Qx - c \|_2`

    Notes
    -----
    - This is an experimental function.
    - If one wishes to perform Lasso or Elastic-Net regression, see
      `sklearn.linear_model.lasso_path <http://scikit-learn.org/stable/modules/
      generated/sklearn.linear_model.lasso_path.html>`_ or
      `sklearn.linear_model.enet_path <http://scikit-learn.org/stable/
      modules/generated/sklearn.linear_model.enet_path.html>`_,
      and pass the parameters `fit_intercept=False, positive=True`


    See Also
    --------
    nonnegative_regression
    """
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

    # NOTE: different from sklearn.linear_model.ridge
    if alpha.size == 1 and n_features > 1:
        alpha = np.repeat(alpha, n_features)

    coef, res = _solve_ridge_nnls(X, y, alpha, solver, **solver_kwargs)

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

    solver : string, optional (default = 'SLSQP')
        Solver with which to solve the QP. Must be one that supports bounds
        (i.e. 'L-BFGS-B', 'TNC', 'SLSQP').

    **solver_kwargs
        See `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/
        reference/generated/scipy.optimize.minimize.html>`_
        for valid keyword arguments


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
    - This is an experimental class.
    - If one wishes to perform Lasso or Elastic-Net regression, see
      `sklearn.linear_model.Lasso <http://scikit-learn.org/stable/modules/
      generated/sklearn.linear_model.Lasso.html>`_ or
      `sklearn.linear_model.ElasticNet <http://scikit-learn.org/stable/
      modules/generated/sklearn.linear_model.ElasticNet.html>`_,
      and pass the parameters `fit_intercept=False, positive=True`

    See Also
    --------
    NonnegativeLinear
    """

    def __init__(self, alpha=1.0, solver='SLSQP', **solver_kwargs):
        if solver not in ('L-BFGS-B', 'TNC', 'SLSQP'):
            raise ValueError('solver must be one of L-BFGS-B, TNC, SLSQP, '
                             'not %s' % solver)
        self.alpha = alpha
        self.solver = solver
        self.solver_kwargs = solver_kwargs

    def fit(self, X, y, sample_weight=None):
        """Fit nonnegative least squares linear model with L2 regularization.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            Training data.

        y : array, shape = (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight : float or array-like, shape (n_samples,), optional (default = None)
            Individual weights for each sample.

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
            X, y, self.alpha, sample_weight=sample_weight,
            solver=self.solver, **self.solver_kwargs)

        return self
