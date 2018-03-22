"""
The :mod:`voxel_model.regressors.nonparametric` module implements ...
"""

# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from .nadaraya_watson import NadarayaWatson, NadarayaWatsonCV
from .polynomial_kernel import Polynomial

__all__ = ['NadarayaWatson',
           'NadarayaWatsonCV',
           'Polynomial']
