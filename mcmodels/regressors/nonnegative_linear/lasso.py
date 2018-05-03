"""
Nonnegative Lasso

:note: This is an experimental module.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

import numpy as np
from sklearn.utils import check_X_y

from .elastic_net import NonnegativeElasticNet, nonnegative_elastic_net_regression


def nonnegative_lasso_regression(X, y, rho, sample_weight=None,
                                 solver='L-BFGS-B', **solver_kwargs):
    r"""Solve the nonnegative least squares estimate lasso regression problem.

    Solves

    .. math::
        \underset{x}{\text{argmin}} \| Ax - b \|_2^2 + \rho \| x \|_1
        \quad \text{s.t.} \quad x \geq 0

    We can write this as the quadratic programming (QP) problem:

    .. math::

        \underset{x}{\text{argmin}} x^TQx - c^Tx \quad \text{s.t.} \quad x \geq 0

    where

    .. math::

        Q = A^TA  \quad \text{and} \quad c = \rho 1^T -2A^Ty

    Parameters
    ----------
    X : array, shape = (n_samples, n_features)
        Training data.

    y : array, shape = (n_samples,) or (n_samples, n_targets)
        Target values.

    rho : float or array with shape = (n_features,)
        Regularization strength; must be a positive float. Improves the
        conditioning of the problem and reduces the variance of the estimates.
        Larger values specify stronger regularization.

    sample_weight : float or array-like, shape (n_samples,), optional (default = None)
        Individual weights for each sample.

    solver : string
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
    This is an experimental function.

    See Also
    --------
    nonnegative_regression
    nonnegative_ridge_regression
    nonnegative_elastic_net_regression
    """
    return nonnegative_elastic_net_regression(
        X, y, rho=rho, sample_weight=sample_weight,
        solver=solver, **solver_kwargs)


class NonnegativeLasso(NonnegativeElasticNet):
    """Nonnegative least squares with L1 regularization.

    This model solves a regression model where the loss function is
    the nonnegative linear least squares function and regularization is
    given by the l1-norm. This estimator has built-in support for
    mulitvariate regression.

    Parameters
    ----------
    rho : float or array with shape = (n_features,)
        Regularization strength; must be a positive float. Improves the
        conditioning of the problem and reduces the variance of the estimates.
        Larger values specify stronger regularization.

    solver : string
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
    >>> from mcmodels.regressors import NonnegativeLasso
    >>> # generate some fake data
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y = np.random.randn(n_samples)
    >>> X = np.random.randn(n_samples, n_features)
    >>> # fit regressor
    >>> reg = NonnegativeLasso(rho=1.0)
    >>> reg.fit(X, y)
    NonnegativeLasso(alpha=1.0)

    Notes
    -----
    This is an experimental class.

    See Also
    --------
    NonnegativeLinear
    NonnegativeRidge
    NonnegativeElasticNet
    """

    def __init__(self, rho=1.0, solver='L-BFGS-B', **solver_kwargs):
        if solver not in ('L-BFGS-B', 'TNC', 'SLSQP'):
            raise ValueError('solver must be one of L-BFGS-B, TNC, SLSQP, '
                             'not %s' % solver)
        self.rho = rho
        self.solver = solver
        self.solver_kwargs = solver_kwargs

    def fit(self, X, y, sample_weight=None):
        """Fit nonnegative least squares linear model with L1 regularization.

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
        self.coef_, self.res_ = nonnegative_lasso_regression(
            X, y, self.rho, sample_weight=sample_weight,
            solver=self.solver, **self.solver_kwargs)

        return self
