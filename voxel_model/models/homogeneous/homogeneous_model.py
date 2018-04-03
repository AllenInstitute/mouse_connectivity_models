"""
Homogeneous Linear Model from Oh et al. 2014.
"""

# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: BSD 3

from sklearn.utils.validation import check_is_fitted

from ...regressors import NonnegativeLinear
from .subset_selection import condition_with_svd_subset_selection


class HomogeneousModel(NonnegativeLinear):

    def __init__(self, kappa=1000, normalize=False, copy_X=True):
        super(HomogeneousModel, self).__init__(normalize=normalize,
                                               copy_X=copy_X)
        self.kappa = kappa

    def fit(self, X, y, sample_weight=None):
        """Fits ...

        """
        X, columns = condition_with_svd_subset_selection(X, self.kappa)
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
