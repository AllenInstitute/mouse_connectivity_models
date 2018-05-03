"""
The :mod:`mcmodels.regressors` module implements scikit-learn style estimators
for solving various regression problems. It implements the NadarayaWatson
regressor, and both regularized and non-regularized non-negative least squares
linear models.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

from .nonnegative_linear import NonnegativeElasticNet
from .nonnegative_linear import NonnegativeLasso
from .nonnegative_linear import NonnegativeLinear
from .nonnegative_linear import NonnegativeRidge
from .nonnegative_linear import nonnegative_elastic_net_regression
from .nonnegative_linear import nonnegative_regression
from .nonnegative_linear import nonnegative_ridge_regression
from .nonnegative_linear import nonnegative_lasso_regression

from . import nonparametric
from .nonparametric import NadarayaWatson
from .nonparametric import NadarayaWatsonCV


__all__ = ['NadarayaWatson',
           'NadarayaWatsonCV',
           'NonnegativeElasticNet',
           'NonnegativeLasso',
           'NonnegativeLinear',
           'NonnegativeRidge',
           'nonnegative_regression',
           'nonnegative_elastic_net_regression',
           'nonnegative_lasso_regression',
           'nonnegative_ridge_regression',
           'nonparametric']
