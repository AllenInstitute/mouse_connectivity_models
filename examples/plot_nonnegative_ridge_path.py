"""
===================================================================
Plot NonnegativeRidge coefficients as a function of regularization
===================================================================

.. note::
    This example is a copy of ``plot_ridge_path.py`` by Fabian Pedregosa
    in the package Scikit-learn, using NonnegativeRidge.

.. currentmodule:: mcmodels.regressors.nonnegative_linear

:class:`NonnegativeRidge` Regression is the estimator used in this example.
Each color represents a different feature of the
coefficient vector, and this is displayed as a function of the
regularization parameter.

This example also shows the usefulness of applying Ridge regression
to highly ill-conditioned matrices. For such matrices, a slight
change in the target variable can cause huge variances in the
calculated weights. In such cases, it is useful to set a certain
regularization (alpha) to reduce this variation (noise).

When alpha is very large, the regularization effect dominates the
squared loss function and the coefficients tend to zero.
At the end of the path, as alpha tends toward zero
and the solution tends towards the ordinary least squares, coefficients
exhibit big oscillations. In practise it is necessary to tune alpha
in such a way that a balance is maintained between both.
"""

# Copied from plot_ridge_path.py by Fabian Pedregosa

# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: BSD 3

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from mcmodels.regressors import NonnegativeRidge

# X is the n  x n Hilbert matrix
n = 6
X = 1. / (np.arange(1, n + 1) + np.arange(n)[:, np.newaxis])
y = np.ones(n)

# #############################################################################
# Compute paths

n_alphas = 550
alphas = np.logspace(-5, 5, n_alphas)

coefs = []
for a in alphas:
    ridge = NonnegativeRidge(alpha=a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

# #############################################################################
# Display results

fig, axes = plt.subplots(1, 2, figsize=(8, 3))

for ax in axes:
    ax.plot(alphas, coefs, lw=2)
    ax.set_xscale('log')
    ax.set_xlabel('alpha')
    ax.set_ylabel('weights')

# trim subplots
axes[0].set_xlim(1e-5, 1e5)
axes[1].set_xlim(1e-1, 1e5)

axes[0].set_ylim(0, 8)
axes[1].set_ylim(0, 1)

plt.suptitle('Ridge coefficients as a function of the regularization')
plt.show()
