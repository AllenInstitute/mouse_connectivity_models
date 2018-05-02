"""
Greedy subset selection for conditioning of input data.
Rank Degeneracy and Least Squares Problems : Golub, Klema, Stewart 1976
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

import numpy as np
import numpy.linalg as LA
import scipy.linalg as linalg
from sklearn.utils import check_random_state


def svd_subset_selection(X, n):
    """svd subset selection to return n cols that are less linearly dependent.

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


def backward_subset_selection_conditioning(X, kappa=1000):
    """Conditioning through subselecting columns of X.

    Uses svd_subset_selection to greedily remove single columns of X till
    the desired conditioning is reached.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Array whose columns we wish to subset.

    kappa : float, optional, default: 1000
        The maximum condition number desired.

    Returns
    -------
    C : array, shape (n_samples, ?)
        Array with condition number < kappa, containing the lesser linearly
        dependent columns of X.
    """
    n_cols = X.shape[1]
    columns = np.arange(n_cols)

    # initialize for first iteration
    subset = np.arange(n_cols)

    while LA.cond(X[:, subset]) > kappa and len(columns) > 1:
        # greedily subset columns of X using svd subset selection
        subset = svd_subset_selection(X, len(columns) - 1)
        columns = columns[subset]

    return X[:, columns], columns


def forward_subset_selection_conditioning(X, kappa=1000, random_state=None):
    """Conditioning through subselecting columns of X.

    Randomly select initial column of X, then greedily add columns that
    minimially increase the conditioning.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Array whose columns we wish to subset.

    kappa : float, optional, default: 1000
        The maximum condition number desired.

    Returns
    -------
    C : array, shape (n_samples, ?)
        Array with condition number < kappa, containing the lesser linearly
        dependent columns of X.
    """
    n_cols = X.shape[1]
    candidates = list(range(n_cols))

    # initial random subset of single column
    initial = check_random_state(random_state).choice(n_cols)

    columns = [initial]
    candidates.remove(initial)

    while LA.cond(np.atleast_2d(X[:, columns])) < kappa and len(columns) < n_cols:
        # greedily subset columns of X using svd subset selection
        condition = [LA.cond(X[:, columns + [c]]) for c in candidates]
        best = candidates[np.argmin(condition)]

        # update
        columns.append(best)
        candidates.remove(best)

    columns = np.sort(columns)

    return X[:, columns], columns
