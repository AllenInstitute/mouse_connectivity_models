.. _nonparametric

====================================
Nonnparametric Regression
====================================

.. current_module:: mcmodels.regressors.nonnparametric

Nonparametric regression is a data-driven alternative to least squares in which
the predictor does not take a predetermined form. Kernel regression estimates
the response by convolving the data with a kernel function to combine the
influence of the individual data points.


Nadaraya Watson Regression
====================================

An example of kernel regression is the Nadaraya-Watson [Nadaraya1010]_
[Watson1010]_ regressor. The Nadaraya-Watson regressor estimates the response
:math:`y` using a kernel weighted average of the individual datapoints
:math:`(x_i, y_i)`:

.. math::
        \hat{y}(x) = \frac{ \sum_i K(x, x_i) Y_i }{ \sum_j K(x, x_j) }

where :math:`K(x, x')` is a kernel function.

Efficient Leave-one-out Cross-Validation
========================================

An advantage of the Nadaraya-Watson regressor is that it allows us to perform
leave-one-out cross validation in a single evaluation of the regressor. This
is useful because often the kernel function :math:`K(x, x')` is parameterized
by a bandwith parameter, say :math:`h` that controls the width of the kernel.
The Nadaraya-Watson regressor can be written as :math:`\hat{y}(x) = \omega Y`,
where:

.. math::
        \omega_i = \frac{ K(x, x_i)  }{ \sum_j K(x, x_j) }

and :math:`Y` is the row stacked array of the responses :math:`y`. For a given
EXPLAIN FURTHER



.. topic:: References

        .. [Nadaraya1010] CITE

        .. [Watson1010] CITE
