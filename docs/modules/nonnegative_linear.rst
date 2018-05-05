.. _nonnegative_linear:

====================================
Nonnegative Least Squares Regression
====================================

.. currentmodule:: mcmodels.regressors

Nonnegative Least Squares Regression solves the equation :math:`Ax=b` subject
to the constraint that the coefficients :math:`x` be nonnegative:

.. math::
        \underset{x}{\text{argmin}} \quad \| Ax - b \|_2^2,
                \quad \text{subject to} \quad x \geq 0

:class:`NonnegativeLinear` will take in its :meth:`~NonnegativeLinear.fit` method
arrays X, y and will store the coefficients :math:`w` in its
:attr:`~NonnegativeLinear.coef_` member:

        >>> from mcmodels.regressors import NonnegativeLinear
        >>> reg = NonnegativeLinear()
        >>> reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
        NonnegativeLinear()
        >>> reg.coef_
        array([1.0, 0.0]


Nonnegative Ridge Regression
----------------------------

The equation :math:`Ax=b` is said to be :term:`ill-conditioned` if the columns of `A`
are nearly linearly dependent. Ill-conditioned least squares problems are highly
sensitive to random errors and produce estimations with high variance as a result.

We can improve the :term:`conditioning` of :math:`Ax=b` by imposing a penalty on the
size of the coefficients :math:`x`. Using the L2 norm as a measure of size, we
arrive at :term:`Tikhonov Regularization`, also known as :term:`ridge regression`:

.. math::
        \underset{x}{\text{argmin}} \| Ax - b \|_2^2 + \alpha^2 \| x \|_2^2

We can incorporate a nonnegativity constraint and rewrite
the formula above as a :term:`quadratic programming` (:term:`QP`) problem:


.. math::
        &\underset{x}{\text{argmin}} \quad
                \| Ax - b \|_2^2 + \alpha^2 \| x \|_2^2
                &\quad \text{s.t.} \quad x \geq 0\\
        &\underset{x}{\text{argmin}} \quad
                (Ax - b)^T(Ax - b) + \alpha^2 (x^T I x)
                &\quad \text{s.t.} \quad x \geq 0\\
        &\underset{x}{\text{argmin}} \quad
                x^TA^TAx - 2b^TAx + b^Tb + x^T \alpha^2 I x
                &\quad  \text{s.t.} \quad  x \geq 0\\
        &\underset{x}{\text{argmin}} \quad
                x^TA^TAx + x^T \alpha^2 I x - 2b^TAx
                &\quad  \text{s.t.} \quad  x \geq 0\\
        &\underset{x}{\text{argmin}} \quad
                x^T( A^TA + \alpha^2 I )x + (- 2b^TA )x
                &\quad  \text{s.t.} \quad  x \geq 0\\
        &\underset{x}{\text{argmin}} \quad
                x^TQx - c^Tx
                &\quad \text{s.t.} \quad x \geq 0
   :label: elastic_net

where

.. math::

        Q = X^TX + \alpha^2 I \quad \text{and} \quad c = - 2A^Ty

which we can solve using any number of :term:`quadratic programming` solvers.


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


Nonnegative Lasso or Nonnegative Elastic Net
--------------------------------------------

Both the non-negative `Lasso <http://scikit-learn.org/stable/modules/
linear_model.html#lasso>`_ and the non-negative `Elastic Net
<http://scikit-learn.org/stable/modules/linear_model.html#elastic-net>`_
regressors are currently implemented in the
`scikit-learn <http://scikit-learn.org/stable/index.html>`_ package:

- If one wishes to perform non-negative :term:`Lasso` regression, see
  `sklearn.linear_model.Lasso <http://scikit-learn.org/stable/modules/
  generated/sklearn.linear_model.Lasso.html>`_ or
  `sklearn.linear_model.lasso_path <http://scikit-learn.org/stable/modules/
  generated/sklearn.linear_model.lasso_path.html>`_
  and pass the parameters `fit_intercept=False, positive=True`
- If one wishes to perform non-negative :term:`Elastic-Net` regression, see
  `sklearn.linear_model.ElasticNet <http://scikit-learn.org/stable/
  modules/generated/sklearn.linear_model.ElasticNet.html>`_, or
  `sklearn.linear_model.enet_path <http://scikit-learn.org/stable/
  modules/generated/sklearn.linear_model.enet_path.html>`_,
  and pass the parameters `fit_intercept=False, positive=True`


.. topic:: Examples

        * :ref:`sphx_glr_auto_examples_plot_nonnegative_ridge_path.py`
        * :ref:`sphx_glr_auto_examples_plot_nonnegative_linear_ridge_variance.py`
