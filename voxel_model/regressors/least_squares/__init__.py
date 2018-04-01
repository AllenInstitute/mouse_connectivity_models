"""
Module for least squares estimators ...
"""

# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: BSD 3

from .scipy_least_squares import ScipyLeastSquares
from .linear import Linear

__all__ = ['Linear',
           'ScipyLeastSquares']
