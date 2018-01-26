from __future__ import division
import mock
import pytest
import numpy as np

@pytest.fixture(scope="session")
def mcc():

    # data
    shape = (10,10,10)

    data_mask = np.ones(shape)
    injection_density = np.ones(shape)
    injection_fraction = np.ones(shape)
    projection_density = np.ones(shape)

    # mock
    mcc = mock.Mock()

    mcc.get_data_mask.return_value = (data_mask, )
    mcc.get_injection_density.return_value = (injection_density, )
    mcc.get_injection_fraction.return_value = (injection_fraction, )
    mcc.get_projection_density.return_value = (projection_density, )

    mcc.get_experiments.return_value = [ {"id":456}, {"id":12}, {"id":315}]

    return mcc
