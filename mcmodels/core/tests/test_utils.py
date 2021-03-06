import pytest

import numpy as np
from numpy.testing \
    import assert_array_equal, assert_array_almost_equal, assert_raises

from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi

from mcmodels.core.utils import compute_centroid, get_injection_hemisphere_id
from mcmodels.core.experiment import _compute_true_injection_density


# ============================================================================
# Module Level Functions
# ============================================================================
def test_compute_centroid():
    # ------------------------------------------------------------------------
    # tests computed centroid is same as MouseConnApi with 'true' inj density
    # pull 'data' from mcc fixture
    a = np.random.rand(4, 4, 4)
    b = np.random.rand(4, 4, 4)

    # compute allensdk centroid
    api = MouseConnectivityApi()
    mcc_centroid = api.calculate_injection_centroid(a, b, 1)

    # 'true' injection density
    _compute_true_injection_density(a, b, inplace=True)

    assert_array_almost_equal(compute_centroid(a), mcc_centroid)


def test_get_injection_hemisphere_id():
    # ------------------------------------------------------------------------
    # tests returns correct hemisphere id
    ones, zeros = np.ones((4, 4)), np.zeros((4, 4))
    left, right = np.dstack((ones, zeros)), np.dstack((zeros, ones))

    assert get_injection_hemisphere_id(left, majority=True) == 1
    assert get_injection_hemisphere_id(right) == 2

    # ------------------------------------------------------------------------
    # tests incompatible input array shape
    assert_raises(ValueError, get_injection_hemisphere_id, np.ones((4,4)))
