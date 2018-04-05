import pytest
import numpy as np
from numpy.testing import assert_raises

from mcmodels.regressors.least_squares import ScipyLeastSquares

class Dummy(ScipyLeastSquares):
    """Dummy class to instatiate ScipyLeastSquares for testing"""

    def _predict(self, coeffs, X):
        """dummy method returns None"""
        return None

# ============================================================================
# ScipyLeastSquares class
# ============================================================================
def test_fit():
    # ------------------------------------------------------------------------
    # tests incompatible sample weight
    X = np.ones((10,10))
    y = np.ones(10)
    reg = Dummy(0)

    assert_raises(ValueError, reg.fit, X, y, np.ones((10,1)))
