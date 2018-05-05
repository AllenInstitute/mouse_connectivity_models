.. _nonparametric:

=========================
Nonnparametric Regression
=========================

.. currentmodule:: mcmodels.regressors

:term:`Nonparametric regression` is a data-driven alternative to least squares in which
the predictor does not take a predetermined form. :term:`Kernel` regression estimates
the response by convolving the data with a kernel function to combine the
influence of the individual data points.


.. _nadaraya_watson:

Nadaraya Watson Regression
==========================

An example of kernel regression is the Nadaraya-Watson
regressor. The Nadaraya-Watson regressor estimates the response
:math:`y` using a kernel weighted average of the individual datapoints
:math:`(x_i, y_i)`:

.. math::
        \hat{y}(x) = \frac{ \sum_i K(x, x_i) Y_i }{ \sum_j K(x, x_j) }

where :math:`K(x, x')` is a kernel function.

.. figure:: ../auto_examples/images/sphx_glr_plot_nadaraya_watson_001.png
        :target: ../auto_examples/plot_nadaraya_watson.html
        :align: center


The model learned by :class:`NadarayaWatson` MORE

Efficient Leave-one-out Cross-Validation
========================================

An advantage of the Nadaraya-Watson regressor is that it allows us to perform
leave-one-out :term:`cross validation` in a single evaluation of the regressor. This
is useful because often the kernel function :math:`K(x, x')` is parameterized
by a bandwidth parameter, say :math:`h` that controls the width of the kernel.
The Nadaraya-Watson regressor can be written as :math:`\hat{y}(x) = \omega Y`,
where:

.. math::
        \omega_i = \frac{ K(x, x_i)  }{ \sum_j K(x, x_j) }

MORE TO COME!!!

.. topic:: Examples

        * :ref:`sphx_glr_auto_examples_plot_nadaraya_watson.py`
