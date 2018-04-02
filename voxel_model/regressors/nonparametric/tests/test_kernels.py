import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from voxel_model.regressors.nonparametric.kernels \
    import _BasePolynomial, Polynomial, Uniform, Epanechnikov, Biweight, Triweight


# ===========================================================================
# _BasePolynomial class
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
    kernel = _BasePolynomial(shape=3)
    X = np.ones((10,10))

    assert_array_equal(X, kernel(X))
    assert_array_equal(kernel(X), kernel(X, X))

    # -----------------------------------------------------------------------
    # test eval_gradient not implemented
    assert_raises(NotImplementedError, kernel, X, eval_gradient=True)


# ===========================================================================
# Polynomial class
# ===========================================================================
def test_polynomial():
    # -----------------------------------------------------------------------
    # test __init__
    polynomial = Polynomial()

    assert hasattr(polynomial, 'shape_bounds')


# ===========================================================================
# Uniform class
# ===========================================================================
def test_uniform():
    # -----------------------------------------------------------------------
    # test convenience class gives proper result
    uniform = Uniform()
    uniform_poly = Polynomial(shape=0)
    X = np.random.rand(10,1)

    assert_array_equal(uniform(X), uniform_poly(X))


# ===========================================================================
# Epanechnikov class
# ===========================================================================
def test_epanechnikov():
    # -----------------------------------------------------------------------
    # test convenience class gives proper result
    epanechnikov = Epanechnikov()
    epanechnikov_poly = Polynomial(shape=1)
    X = np.random.rand(10,1)

    assert_array_equal(epanechnikov(X), epanechnikov_poly(X))


# ===========================================================================
# Biweight class
# ===========================================================================
def test_biweight():
    # -----------------------------------------------------------------------
    # test convenience class gives proper result
    biweight = Biweight()
    biweight_poly = Polynomial(shape=2)
    X = np.random.rand(10,1)

    assert_array_equal(biweight(X), biweight_poly(X))


# ===========================================================================
# Triweight class
# ===========================================================================
def test_triweight():
    # -----------------------------------------------------------------------
    # test convenience class gives proper result
    triweight = Triweight()
    triweight_poly = Polynomial(shape=3)
    X = np.random.rand(10,1)

    assert_array_equal(triweight(X), triweight_poly(X))
