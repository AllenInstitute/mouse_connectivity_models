import pytest
import numpy as np
from numpy.testing import assert_raises

from voxel_model.regressors.nonparametric.kernels import Polynomial


# ===========================================================================
# Polynomial (kernel) class
# ===========================================================================

def test_coefficient():
    # -----------------------------------------------------------------------
    # test coefficient property correct
    uniform = Polynomial(shape=0)
    epanechnikov = Polynomial(shape=1)
    biweight = Polynomial(shape=2)
    biweight_supp = Polynomial(shape=2, support=2)

    assert uniform.coefficient == pytest.approx(0.5)
    assert epanechnikov.coefficient == pytest.approx(0.75)
    assert biweight.coefficient == pytest.approx(0.9375)
    assert biweight_supp.coefficient == pytest.approx(0.5*0.9375)


def test_call():
    # -----------------------------------------------------------------------
    # test __call__ returns what we expect

    # -----------------------------------------------------------------------
    # test eval_gradient not implemented

    pass
