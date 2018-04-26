"""
The :mod:`voxel_model.models.homogeneous` module implements the Homogeneous
Model for predicting regional connectivity from Oh et al. 2014. Additionally,
the module implements a greedy conditioning procedure based on the singular
value decomposition and QR pivoting.
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
