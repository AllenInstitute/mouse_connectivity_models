"""
Linear Function for ...
"""

# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: BSD 3

import numpy as np

from .scipy_least_squares import ScipyLeastSquares


class Linear(ScipyLeastSquares):
    """Fits linear least squares model using ScipyLeastSquares.

    Fits the model:
        `` y = \beta_0 + \beta_1 x ``

    where ``\beta_0`` and ``\beta_1`` are scalars.

    Parameters
    ----------
    x0 : array shape (2,), optional, default np.array([0,0])
        Initial guesses for ``\beta_0`` and ``\beta_1``

    **least_squares_kwargs :
        Keyword arguments to be provided to scipy.optimize.least_squares.

        Options include:
        ----------------
            bounds
            method
            ftol
            xtol
            gtol
            x_scale
            loss
            f_scale
            max_nfev
            diff_step
            tr_solver
            tr_options

    See also
    --------
    :class:`ScipyLeastSquare`, scipy.optimize.least_squares

    Examples
    --------
    >>> import numpy as np
    >>> from voxel_model.regressors.least_squares import Linear
    >>> X = np.array([-2, 1, 4])
    >>> y = np.array([4.9, 1.3, -2.3])
    >>> reg = Linear()
    >>> reg.fit(X, y)
    >>> reg.opt_result_.x
    np.array([2.5, -1.2])
    """

    DEFUALT_X0 = np.zeros(2)

    def __init__(self, x0=None, **least_squares_kwargs):
        """Allow for optional x0"""
        if x0 is None:
            # initialize coefficients @ 0
            x0 = Linear.DEFUALT_X0

        super(Linear, self).__init__(x0, **least_squares_kwargs)

    def _predict(self, coeffs, X):
        return coeffs[0] + coeffs[1]*X
