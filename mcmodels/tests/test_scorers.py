import pytest
import numpy as np
from numpy.testing import assert_raises

from mcmodels.scorers \
    import (_unionize, _voxelize, mean_squared_relative_error,
            regional_mean_squared_relative_error, voxel_mean_squared_relative_error,
            mse_rel, regional_mse_rel, voxel_mse_rel)

def test_unionize():
    # ------------------------------------------------------------------------
    # test correct output
    v = np.random.rand(10, 8)
    ipsi_key = [0, 0, 0, 0, 1, 1, 2, 0]
    contra_key = [2, 3, 1, 1, 0, 0, 0, 0]

    # TODO

    # ------------------------------------------------------------------------
    # test key incompatibility with each other
    ipsi_key = np.ones(4)

    assert_raises(ValueError, _unionize, v, ipsi_key, contra_key)

    # ------------------------------------------------------------------------
    # test key incompatibility with v
    contra_key = np.ones(4)

    assert_raises(ValueError, _unionize, v, ipsi_key, contra_key)


def test_voxelize():
    # ------------------------------------------------------------------------
    # test correct output
    v = np.random.rand(10, 8)
    ipsi_key = [0, 0, 0, 0, 1, 1, 2, 0]
    contra_key = [2, 3, 1, 1, 0, 0, 0, 0]

    # TODO

    # ------------------------------------------------------------------------
    # test key incompatibility with each other
    ipsi_key = np.ones(4)

    assert_raises(ValueError, _voxelize, v, ipsi_key, contra_key)

    # ------------------------------------------------------------------------
    # test key incompatibility with v
    contra_key = np.ones(4)

    assert_raises(ValueError, _voxelize, v, ipsi_key, contra_key)


def test_mean_squared_relative_error():
    # ------------------------------------------------------------------------
    # test correct output
    # TODO
    pass


def test_regional_mean_squared_relative_error():
    # ------------------------------------------------------------------------
    # test correct output
    # TODO
    pass


def test_voxel_mean_squared_relative_error():
    # ------------------------------------------------------------------------
    # test correct output
    # TODO
    pass
