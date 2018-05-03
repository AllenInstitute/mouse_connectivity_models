"""
===================================================================
Plot NonnegativeLasso coefficients as a function of regularization
===================================================================

.. note::
    This example is a copy of ``plot_ridge_path.py`` by Fabian Pedregosa
    in the package Scikit-learn, using NonnegativeLasso.

.. currentmodule:: mcmodels.regressors.nonnegative_linear

"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

# NOTE: modified from plot_ridge_path.py by Fabian Pedregosa
#       from the package Scikit-Learn licensed under the 3 clause BSD License
#       reproduced below:
#
# New BSD License
#
# Copyright (c) 2007–2018 The scikit-learn developers.
# All rights reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of the Scikit-learn Developers  nor the names of
#      its contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.


from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

from mcmodels.regressors import NonnegativeLasso


print(__doc__)

# X is the n  x n Hilbert matrix
n = 6
X = 1. / (np.arange(1, n + 1) + np.arange(n)[:, np.newaxis])
y = np.ones(n)

# #############################################################################
# Compute paths

n_rhos = 31
rhos = np.logspace(-2, 1, n_rhos)

coefs = []
for r in rhos:
    lasso = NonnegativeLasso(rho=r)
    lasso.fit(X, y)
    coefs.append(lasso.coef_)

# #############################################################################
# Display results

fig, axes = plt.subplots(1, 2, figsize=(8, 3))

for ax in axes:
    ax.plot(rhos, coefs, lw=2)
    ax.set_xscale('log')
    ax.set_xlabel('rho')
    ax.set_ylabel('weights')

# trim subplots
axes[0].set_xlim(1e-2, 1e1)
axes[1].set_xlim(1e-1, 1e1)

axes[0].set_ylim(0, 8)
axes[1].set_ylim(0, 3)

plt.suptitle('Lasso coefficients as a function of the regularization')
plt.show()
