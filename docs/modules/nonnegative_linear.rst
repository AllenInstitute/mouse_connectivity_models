.. _nonnegative_linear:

====================================
Nonnegative Least Squares Regression
====================================

.. currentmodule:: mcmodels.regressors.nonnegative_linear

Nonnegative Least Squares Regression solves the equation :math:`Ax=b` subject
to the constraint that the coefficients :math:`x` be nonnegative:

.. math::
        \underset{x}{\text{argmin}} \| Ax - b \|_2^2, \text{subject to} x \geq 0

:class:`NonnegativeLinear` will take in its ``fit`` method arrays X, y and
will store the coefficients :math:`w` in its ``coef_`` member:

        >>> from mcmodels.regressors import NonnegativeLinear
        >>> reg = NonnegativeLinear()
        >>> reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
        NonnegativeLinear()
        >>> reg.coef_
        array([0.5, 0.5]

However, CONDITIONING


satisfied the above constraint. We enforce the coefficients to be positive as
a weight between brain regions must be positive.


Nonnegative Ridge Regression
============================

The equation :math:`Ax=b` is said to be ill-conditioned if the columns of `A`
are nearly linearly dependent. Ill-conditioned least squares problems are highly
sensitive to random errors and produce estimations with high variance as a result.

We can improve the conditioning of :math:`Ax=b` by imposing a penalty on the
size of the coefficients :math:`x`. Using the L2 norm as a measure of size, we
arrive at Tikhonov Reglarization, also known as ridge regression:

.. math::
        \text{argmin}_x \| Ax - b \|_2^2 + \alpha^2 \| x \|_2^2

where :math:`\alpha` is the penalty vector. We can rewrite this with a
nonnegativity constraint as:

.. math::
        \text{argmin}_x \| Qx - c \|_2^2, \text{subject to} x \geq 0

where

.. math::
        Q = A^T A + \alpha I \quad \text{and} \quad c = A^T b

.. figure:: ../auto_examples/images/sphx_glr_plot_nonnegative_ridge_path_001.png
        :target: ../auto_examples/plot_nonnegative_ridge_path.html
        :align: center
        :scale: 50%

As with :class:`NonnegativeLinear`, :class:`NonnegativeRidge`  will take in its
``fit`` method arrays X, y and will store the coefficients :math:`w` in its
``coef_`` member:

        >>> from mcmodels.regressors import NonnegativeRidge
        >>> reg = NonnegativeRidge(alpha=1.0)
        >>> reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
        NonnegativeRidge(alpha=1.0)
        >>> reg.coef_


.. topic:: Examples

        * :ref:`sphx_glr_auto_examples_plot_nonnegative_ridge_path.py`
        * :ref:`sphx_glr_auto_examples_plot_nonnegative_linear_ridge_variance.py`
