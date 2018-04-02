"""
Homogeneous Linear Model from Oh et al. 2014.
"""
import numpy as np
import numpy.linalg as LA
import scipy.linalg as linalg

from sklearn.utils.validation import check_is_fitted

from ..regressors import NonnegativeLinear


def svd_subset_selection(X, kappa=1000):
    """greedily subset X to have good conditioning"""
    def svd_subset(x, n):
        """svd subset selection to return n cols that ~less lin dependent"""
        if n > x.shape[1]:
            raise ValueError("n cannot be greater than the number of columns of x")
        _, _, vh  = linalg.svd(x, full_matrices=False, compute_uv=True)
        _, _, p = linalg.qr(vh[:n], pivoting=True)
        return p[:n]

    # NOTE: may want sklearn.utils.check_X_y
    X_ = np.atleast_2d(X.copy())

    columns_ = np.arange(X.shape[1])
    while LA.cond(X_):
        if len(X_.shape) == 1:
            raise ValueError("Cannot condition matrix well enough")

        subset = svd_subset(X_, X_.shape[1] - 1)
        X_ = X_[:, subset]
        columns_ = columns_[:, subset]

    return X_, columns_


class HomogeneousModel(NonnegativeLinear):

    def __init__(self, kappa=1000, normalize=False, copy_X=True):
        super(HomogeneousModel, self).__init__(normalize=normalize,
                                               copy_X=copy_X)
        self.kappa = kappa

    def fit(self, X, y, sample_weight=None):
        """Fits ...

        """
        X_, columns_ = svd_subset_selection(X, kappa=self.kappa)
        self.columns_ = columns_

        super(HomogeneousModel, self).fit(X_, y, sample_weight=sample_weight)

    def predict(self, X):
        """Predicts ...

        """
        check_is_fitted(self, "columns_")

        super(HomogeneousModel, self).predict(X[:, self.columns_])


    @property
    def weights(self):
        """Convenience property for pulling out regional matrix."""
        check_is_fitted(self, ["coef_"])
        return self.coef_.T
