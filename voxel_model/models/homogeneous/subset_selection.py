"""
Greedy subset selection ...
"""

# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: BSD 3

import numpy as np
import numpy.linalg as LA
import scipy.linalg as linalg


def svd_subset_selection(X, n):
    """svd subset selection to return n cols that ~less lin dependent"""
    # NOTE: may want sklearn.utils.check_X_y
    X = np.atleast_2d(X.copy())

    if n > X.shape[1]:
        raise ValueError("n must be less than the number of columns of X")
    if n < 1:
        raise ValueError("n must be at least 1")

    _, _, vh  = linalg.svd(X, full_matrices=False, compute_uv=True)
    _, _, p = linalg.qr(vh[:n], pivoting=True)
    return p[:n]


def condition_with_svd_subset_selection(X, kappa=1000):
    """greedily subset X to have good conditioning"""
    X = np.atleast_2d(X.copy())
    columns = np.arange(X.shape[1])

    while LA.cond(X) > kappa and X.shape[1] > 1:
        # greedily subset columns of X using svd subset selection
        subset = svd_subset_selection(X, X.shape[1] - 1)
        X = X[:, subset]
        columns = columns[:, subset]

    return X, columns
