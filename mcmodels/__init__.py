"""
Python package providing mesoscale connectivity models for mouse
================================================================

mcmodels is a Python package containing the models for mesoscale connectivty
in mouse developed at the Allen Institute for Brain Science.

See https://AllenInstitute.githib.io/mouse_connectivity_models for complete
documentation.
"""

# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: BSD 3

__version__ = '0.0.9'

from . import core
from . import models
from . import regressors
from . import utils

__all__ = ['core',
           'models',
           'regressors',
           'utils']
