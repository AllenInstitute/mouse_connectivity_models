"""
Greedy subset selection for conditioning of input data.
Rank Degeneracy and Least Squares Problems : Golub, Klema, Stewart 1976
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

import numpy as np
import numpy.linalg as LA
import scipy.linalg as linalg


def svd_subset_selection(X, n):
    """svd subset selection to return n cols that ~less lin dependent.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Array whose columns we wish to subset.

    n : int
        The number of columns to return.

    Returns
    -------
    C : array, shape (n_samples, n)
        Array containing the n ~least dependent columns of X.
    """
    # NOTE: may want sklearn.utils.check_X_y
    X = np.atleast_2d(X.copy())

    if n > X.shape[1]:
        raise ValueError("n must be less than the number of columns of X")
    if n < 1:
        raise ValueError("n must be at least 1")

    _, _, vh  = linalg.svd(X, full_matrices=False)
    _, _, p = linalg.qr(vh[:n], pivoting=True)
    return p[:n]


def condition_with_svd_subset_selection(X, kappa=1000):
    """Conditioning through subselecting columns of X.

    Uses svd_subset_selection to greedily remove single columns of X till
    the desired conditioning is reached

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Array whose columns we wish to subset.

    kappa : float, optional, default: 1000
        The maximum condition number desired.

    Returns
    -------
    C : array, shape (n_samples, ?)
        Array with condition number < kappa, containing the ~lesser dependent
        columns of X.
    """
    X = np.atleast_2d(X.copy())
    columns = np.arange(X.shape[1])

    while LA.cond(X) > kappa and X.shape[1] > 1:
        # greedily subset columns of X using svd subset selection
        subset = svd_subset_selection(X, X.shape[1] - 1)
        X = X[:, subset]
        columns = columns[:, subset]

    return X, columns
