"""
The :mod:`regressors` module implements scikit-learn style estimators
for solving various regression problems. It implements the NadarayaWatson
regressor, a general least squares regressor, and both regularized and
non-regularized Nonnegative least squares models.
"""

# Authors: Joseph Knox <josephk@alleninstitute.org>
# Licence: BSD 3

from . import least_squares

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
           'least_squares',
           'nonnegative_regression',
           'nonnegative_ridge_regression',
           'nonparametric']
