import pytest
import numpy as np
from numpy.testing import assert_array_equal

from voxel_model.regressors.least_squares import Linear

# ----------------------------------------------------------------------------
# tests
def test_predict():
    coeffs = 2.5, -1.2
    X = np.array([-2, 1, 4])
    y = np.array([4.9, 1.3, -2.3])

    y_pred = Linear()._predict(coeffs, X)

    assert_array_equal(y, y_pred)
