"""
Homogeneous Linear Model from Oh et al. 2014.
"""

# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: BSD 3

import numpy as np
import numpy.linalg as LA
import scipy.linalg as linalg

from sklearn.utils.validation import check_is_fitted

from ..regressors import NonnegativeLinear


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


class HomogeneousModel(NonnegativeLinear):

    def __init__(self, kappa=1000, normalize=False, copy_X=True):
        super(HomogeneousModel, self).__init__(normalize=normalize,
                                               copy_X=copy_X)
        self.kappa = kappa

    def _condition_X(self, X):
        """greedily subset X to have good conditioning"""
        # NOTE: may want sklearn.utils.check_X_y
        X = np.atleast_2d(X.copy())
        columns = np.arange(X.shape[1])

        while LA.cond(X) > self.kappa and X.shape[1] > 1:
            # greedily subset columns of X using svd subset selection
            subset = svd_subset_selection(X, X.shape[1] - 1)
            X = X[:, subset]
            columns = columns[:, subset]

        return X, columns

    def fit(self, X, y, sample_weight=None):
        """Fits ...

        """
        X, columns = self._condition_X(X)
        self.columns_ = columns

        super(HomogeneousModel, self).fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        """Predicts ...

        """
        check_is_fitted(self, "columns_")
        y_pred = super(HomogeneousModel, self).predict(X[:, self.columns_])

        return y_pred

    @property
    def weights(self):
        """Convenience property for pulling out regional matrix."""
        check_is_fitted(self, ["coef_"])
        return self.coef_.T
