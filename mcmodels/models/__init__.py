"""
The :mod:`voxel_model.models` module implements models that have been developed
here at the Allen Institute for modeling mesoscale connectivity in the mouse.
The module contains the HomogeneousModel from Oh et al. 2014 as well as the
recent VoxelModel from Knox & Harris 2018.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

from . import homogeneous
from .homogeneous import HomogeneousModel

from . import voxel
from .voxel import VoxelModel

__all__ = ['HomogeneousModel',
           'VoxelModel',
           'homogeneous',
           'voxel']
