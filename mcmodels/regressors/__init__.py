"""
The :mod:`mcmodels.regressors` module implements scikit-learn style estimators
for solving various regression problems. It implements the NadarayaWatson
regressor, a general least squares regressor, and both regularized and
non-regularized Nonnegative least squares models.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

from .nonnegative_linear import NonnegativeLinear
from .nonnegative_linear import NonnegativeRidge
from .nonnegative_linear import nonnegative_regression
from .nonnegative_linear import nonnegative_ridge_regression

from . import nonparametric
from .nonparametric import NadarayaWatson
from .nonparametric import NadarayaWatsonCV


__all__ = ['NadarayaWatson',
           'NadarayaWatsonCV',
           'NonnegativeLinear',
           'NonnegativeRidge',
           'nonnegative_regression',
           'nonnegative_ridge_regression',
           'nonparametric']
