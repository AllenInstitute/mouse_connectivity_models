"""
The :mod:`mcmodels.regressors.nonnegative_linear` module implements linear
models subject to the nonnegativity constraint. It includes Nonnegative linear
regression and experimental modules implementing Nonnegative linear regression
with L2 (Ridge) regularization.

.. note::
    - If one wishes to perform non-negative Lasso regression, see
      `sklearn.linear_model.Lasso <http://scikit-learn.org/stable/modules/
      generated/sklearn.linear_model.Lasso.html>`_ or
      `sklearn.linear_model.lasso_path <http://scikit-learn.org/stable/modules/
      generated/sklearn.linear_model.lasso_path.html>`_
      and pass the parameters `fit_intercept=False, positive=True`
    - If one wishes to perform non-negative Elastic-Net regression, see
      `sklearn.linear_model.ElasticNet <http://scikit-learn.org/stable/
      modules/generated/sklearn.linear_model.ElasticNet.html>`_, or
      `sklearn.linear_model.enet_path <http://scikit-learn.org/stable/
      modules/generated/sklearn.linear_model.enet_path.html>`_,
      and pass the parameters `fit_intercept=False, positive=True`
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

from .base import NonnegativeLinear
from .base import nonnegative_regression

from .ridge import NonnegativeRidge
from .ridge import nonnegative_ridge_regression


__all__ = ['NonnegativeLinear',
           'NonnegativeRidge',
           'nonnegative_regression',
           'nonnegative_ridge_regression']
