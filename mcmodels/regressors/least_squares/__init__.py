"""
The :mod:`voxel_model.regressors.least_squares` module implements models that
utilize scipy.optimize.least_squares as a backend. It includes ScipyLeastSquares,
a metaclass that wraps scipy.optimize.least_squares as a scikit-learn estimator
and Linear, an implementation of a simple scalar linear regression with a
single coefficient and an intercept.
"""

# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: BSD 3

from .scipy_least_squares import ScipyLeastSquares
from .linear import Linear

__all__ = ['Linear',
           'ScipyLeastSquares']
