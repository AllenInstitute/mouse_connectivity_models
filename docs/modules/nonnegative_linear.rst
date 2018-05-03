.. _nonnegative_linear:

====================================
Nonnegative Least Squares Regression
====================================

.. currentmodule:: mcmodels.regressors

Nonnegative Least Squares Regression solves the equation :math:`Ax=b` subject
to the constraint that the coefficients :math:`x` be nonnegative:

.. math::
        \underset{x}{\text{argmin}} \| Ax - b \|_2^2, \text{subject to } x \geq 0

:class:`NonnegativeLinear` will take in its :meth:`~NonnegativeLinear.fit` method
arrays X, y and will store the coefficients :math:`w` in its
:attr:`~NonnegativeLinear.coef_` member:

        >>> from mcmodels.regressors import NonnegativeLinear
        >>> reg = NonnegativeLinear()
        >>> reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
        NonnegativeLinear()
        >>> reg.coef_
        array([1.0, 0.0]

However, CONDITIONING

satisfied the above constraint. We enforce the coefficients to be positive as
a weight between brain regions must be positive.


Nonnegative Ridge Regression
----------------------------

The equation :math:`Ax=b` is said to be ill-conditioned if the columns of `A`
are nearly linearly dependent. Ill-conditioned least squares problems are highly
sensitive to random errors and produce estimations with high variance as a result.

We can improve the conditioning of :math:`Ax=b` by imposing a penalty on the
size of the coefficients :math:`x`. Using the L2 norm as a measure of size, we
arrive at Tikhonov Reglarization, also known as ridge regression:

.. math::
        \underset{x}{\text{argmin}} \| Ax - b \|_2^2 + \alpha^2 \| x \|_2^2

Using the derivation below :eq:`elastic_net`, we can incoporate a nonnegativity
constraint and write the non-negative ridge regression problem as a :term:`quadratic
programing` (:term:`QP`) problem:

.. math::
        \underset{x}{\text{argmin}} \quad
                x^T(A^TA + \alpha^2 I)x + (-2A^Tb)^Tx
                \quad \text{s.t.} \quad x \geq 0


.. figure:: ../auto_examples/images/sphx_glr_plot_nonnegative_ridge_path_001.png
        :target: ../auto_examples/plot_nonnegative_ridge_path.html
        :align: center

As with :class:`NonnegativeLinear`, :class:`NonnegativeRidge`  will take in its
:meth:`~NonnegativeRidge.fit` method arrays X, y and will store the coefficients
:math:`w` in its :attr:`~NonnegativeRidge.coef_` member:

        >>> from mcmodels.regressors import NonnegativeRidge
        >>> reg = NonnegativeRidge(alpha=1.0)
        >>> reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
        NonnegativeRidge(alpha=1.0, solver='SLSQP')
        >>> reg.coef_
        array([0.45454545, 0.45454545])


Nonnegative Lasso
-----------------

Ridge Regression NO SPARSITY

.. math::
        \text{argmin}_x \| Ax - b \|_2^2 + \rho \| x \|_1

Using the above derivation, we can incoporate a nonnegativity constraint and
write as a QP:

.. math::
        \underset{x}{\text{argmin}} \quad
                x^T(A^TA)x + (\rho 1^T - 2A^Tb)^Tx
                \quad \text{s.t.} \quad x \geq 0

.. figure:: ../auto_examples/images/sphx_glr_plot_nonnegative_lasso_path_001.png
        :target: ../auto_examples/plot_nonnegative_lasso_path.html
        :align: center

As with :class:`NonnegativeLinear`, :class:`NonnegativeLasso`  will take in its
:meth:`~NonnegativeLasso.fit` method arrays X, y and will store the coefficients
:math:`w` in its :attr:`~NonnegativeLasso.coef_` member:

        >>> from mcmodels.regressors import NonnegativeLasso
        >>> reg = NonnegativeLasso(rho=1.0)
        >>> reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
        NonnegativeLasso(alpha=1.0, solver='SLSQP')
        >>> reg.coef_
        array([0.45, 0.45])


Nonnegative Elastic Net
-----------------------

Elastic net is a combination of L2 and L1 regularization. Since :math:`x\geq0`,
we have :math:`\|x\|_1 = \sum_i x_i = 1^T x`, which we use to derive the elastic
net (and consequently ridge/Lasso) regularized nonnegative linear regression:

.. math::
        &\underset{x}{\text{argmin}} \quad
                \| Ax - b \|_2^2 + \alpha^2 \| x \|_2^2 + \rho \| x \|_1
                &\quad \text{s.t.} \quad x \geq 0\\
        &\underset{x}{\text{argmin}} \quad
                (Ax - b)^T(Ax - b) + \alpha^2 (x^T I x) + \rho (1^T x)
                &\quad \text{s.t.} \quad x \geq 0\\
        &\underset{x}{\text{argmin}} \quad
                x^TA^TAx - 2b^TAx + b^Tb + x^T \alpha^2 I x + \rho 1^T x
                &\quad  \text{s.t.} \quad  x \geq 0\\
        &\underset{x}{\text{argmin}} \quad
                x^TA^TAx + x^T \alpha^2 I x + \rho 1^T x - 2b^TAx
                &\quad  \text{s.t.} \quad  x \geq 0\\
        &\underset{x}{\text{argmin}} \quad
                x^T( A^TA + \alpha^2 I )x + (\rho 1^T - 2b^TA )x
                &\quad  \text{s.t.} \quad  x \geq 0\\
        &\underset{x}{\text{argmin}} \quad
                x^TQx - c^Tx
                &\quad \text{s.t.} \quad x \geq 0
   :label: elastic_net

where

.. math::

        Q = X^TX + \alpha^2 I \quad \text{and} \quad c = \rho 1^T - 2A^Ty

which we can solve using any number of quadratic programming solvers.

Thus for the non-negative Lasso regularized estimate, we set :math:`\alpha=0`,
and for the non-negative ridge regularized estimate, we set :math:`\rho=0`.

As with :class:`NonnegativeLinear`, :class:`NonnegativeElasticNet`  will take in its
:meth:`~NonnegativeElasticNet.fit` method arrays X, y and will store the coefficients
:math:`w` in its :attr:`~NonnegativeElasticNet.coef_` member:

        >>> from mcmodels.regressors import NonnegativeElasticNet
        >>> reg = NonnegativeElasticNet(alpha=1.0, rho=1.0)
        >>> reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
        NonnegativeElasticNet(alpha=1.0, rho=1.0, solver='SLSQP')
        >>> reg.coef_
        array([0.45454545, 0.45454545])


.. topic:: Examples

        * :ref:`sphx_glr_auto_examples_plot_nonnegative_ridge_path.py`
        * :ref:`sphx_glr_auto_examples_plot_nonnegative_lasso_path.py`
        * :ref:`sphx_glr_auto_examples_plot_nonnegative_linear_ridge_variance.py`
