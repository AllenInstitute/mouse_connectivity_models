"""
Nonnegative Ridge Regression.

:note: This is an experimental module.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

import numpy as np
from sklearn.utils import check_X_y

from .elastic_net import NonnegativeElasticNet, nonnegative_elastic_net_regression


def nonnegative_ridge_regression(X, y, alpha, sample_weight=None,
                                 solver='L-BFGS-B', **solver_kwargs):
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
    alpha = np.asarray(alpha, dtype=X.dtype).ravel()
    rho = np.zeros_like(alpha) # for compatibility

    return nonnegative_elastic_net_regression(
        X, y, alpha, rho, sample_weight=sample_weight,
        solver=solver, **solver_kwargs)


class NonnegativeRidge(NonnegativeElasticNet):
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

    def __init__(self, alpha=1.0, solver='L-BFGS-B', **solver_kwargs):
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
