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
    rho = np.asarray(rho, dtype=X.dtype).ravel()
    alpha = np.zeros_like(rho) # for compatibility

    return nonnegative_elastic_net_regression(
        X, y, alpha, rho, sample_weight=sample_weight,
        solver=solver, **solver_kwargs)


class NonnegativeLasso(NonnegativeElasticNet):

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
