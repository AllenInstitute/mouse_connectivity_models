"""
The :mod:`mcmodels.models.homogeneous` module implements a Homogeneous
Model for predicting regional connectivity similar to Oh et al. 2014.
Additionally, the module implements greedy forward/backward subset selection
algorithms to improve the conditioning of given input arrays.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

from .homogeneous_model import HomogeneousModel

from .subset_selection import svd_subset_selection
from .subset_selection import forward_subset_selection_conditioning
from .subset_selection import backward_subset_selection_conditioning


__all__ = ['HomogeneousModel',
           'backward_subset_selection_conditioning',
           'forward_subset_selection_conditioning',
           'svd_subset_selection']
