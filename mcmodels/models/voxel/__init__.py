"""
The :mod:`voxel_model.models.voxel` module implements the voxel scale model
or predicting voxel-scale connectivity from Knox & Harris et al. 2018.
The module also contains a VoxelConnectivityArray class to work with the model
in memory (implicitly computing its weights) and a RegionalizedModel class to
integrate the model weights to the regional level.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

from .regionalized_model import RegionalizedModel
from .voxel_connectivity_array import VoxelConnectivityArray
from .voxel_model import VoxelModel

__all__ = ['RegionalizedModel',
           'VoxelConnectivityArray',
           'VoxelModel']
