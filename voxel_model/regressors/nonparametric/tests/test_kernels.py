import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from voxel_model.regressors.nonparametric.kernels \
    import _BasePolynomial, Polynomial, Uniform, Epanechnikov, Biweight, Triweight


# ===========================================================================
# _BasePolynomial (kernel) class
# ===========================================================================
def test_coefficient():
    # -----------------------------------------------------------------------
    # test coefficient property correct
    uniform = _BasePolynomial(shape=0)
    epanechnikov = _BasePolynomial(shape=1)
    biweight = _BasePolynomial(shape=2)
    biweight_supp = _BasePolynomial(shape=2, support=2)

    assert uniform.coefficient == pytest.approx(0.5)
    assert epanechnikov.coefficient == pytest.approx(0.75)
    assert biweight.coefficient == pytest.approx(0.9375)
    assert biweight_supp.coefficient == pytest.approx(0.5*0.9375)


def test_call():
    # -----------------------------------------------------------------------
    # test __call__ returns what we expect
    # TODO

    # -----------------------------------------------------------------------
    # test eval_gradient not implemented
    kernel = _BasePolynomial(shape=3)
    X = np.ones((10,10))

    assert_raises(NotImplementedError, kernel, X, eval_gradient=True)
