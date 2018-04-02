"""
The :mod:`voxel_model.regressors` module includes ...
"""

from . import least_squares

from .nonnegative_linear import NonnegativeLinear
from .nonnegative_linear import NonnegativeRidge
from .nonnegative_linear import nonnegative_regression
from .nonnegative_linear import nonnegative_ridge_regression

from . import nonparametric
from .nonparametric import NadarayaWatson
from .nonparametric import NadarayaWatsonCV

from . import scorers

__all__ = ['NadarayaWatson',
           'NadarayaWatsonCV',
           'NonnegativeLinear',
           'NonnegativeRidge',
           'least_squares',
           'nonnegative_regression',
           'nonnegative_ridge_regression',
           'nonparametric',
           'scorers']
