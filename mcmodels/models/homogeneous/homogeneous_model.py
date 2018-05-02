"""
Homogeneous Linear Model similar to Oh et al. 2014.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

from sklearn.utils.validation import check_is_fitted

from ...regressors import NonnegativeLinear
from .subset_selection import backward_subset_selection_conditioning


class HomogeneousModel(NonnegativeLinear):
    """Homogeneous model similar to Oh et al. 2014.

    Implements the Homogeneous model for fitting nonnegative weights at the
    regional level. There is an additional constraint on the features to
    ensure the regression is well conditioned.

    Parameters
    ----------
    kappa : float, optional, default: 1000
        The maximum condition number allowed for input data arrays.

    Attributes
    ----------
    columns_ : array
        The features that are included in the regression after the conditioning
        has occurred.

    coef_ : array, shape (n_targets, n_features)
        The model weights.

    Examples
    --------
    >>> from mcmodels.core import VoxelModelCache, RegionalData
    >>> from mcmodels.models import HomogeneousModel
    >>> # get data with whcich to fit model
    >>> cache = VoxelModelCache()
    >>> voxel_data = cache.get_experiment_data()
    >>> regional_data = RegionalData.from_voxel_data(voxel_data)
    >>> # fit model
    >>> reg = HomogeneousModel()
    >>> reg.fit(regional_data.injections, regional_data.projections)
    HomogeneousModel(kappa=1000)

    References
    ----------
    Oh et al. 2014. A mesoscale connectome of the mouse brain. Nature,
        508(7495), 207-214. doi: 10.1038/nature13186
    """

    def __init__(self, kappa=1000):
        self.kappa = kappa

    def fit(self, X, y, sample_weight=None):
        """Fit HomogeneousModel.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.

        y : array, shape (n_samples, n_features)
            Target values.

        Returns
        -------
        self : returns an instance of self
        """
        X, columns = backward_subset_selection_conditioning(X, self.kappa)
        self.columns_ = columns

        return super(HomogeneousModel, self).fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        """Predict using the HomogeneousModel.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        C : array, shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        check_is_fitted(self, "columns_")
        y_pred = super(HomogeneousModel, self).predict(X[:, self.columns_])

        return y_pred

    @property
    def weights(self):
        """Convenience property for pulling out regional matrix."""
        check_is_fitted(self, ["coef_"])
        return self.coef_.T
