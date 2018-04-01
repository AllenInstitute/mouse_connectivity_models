"""
Linear Function for ...
"""

# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: BSD 3

import numpy as np

from .scipy_least_squares import ScipyLeastSquares


class Linear(ScipyLeastSquares):
    """Fits linear using ScipyLeastSquares"""

    DEFUALT_X0 = np.zeros(2)

    def __init__(self, x0=None, **least_squares_kwargs):
        """Allow for optional x0"""
        if x0 is None:
            # initialize coefficients @ 0
            x0 = Linear.DEFUALT_X0

        super(Linear, self).__init__(x0, **least_squares_kwargs)

    def _predict(self, coeffs, X):
        return coeffs[0] + coeffs[1]*X
