# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import division
from functools import partial, reduce
import os
import numpy as np
import operator as op

class VoxelModelArray(object):
    """Class for implicit construction of the voxel model.

    See voxel_model.VoxelModel for weights/nodes descriptions

    Parameters
    ----------
    weights : array-like, shape (n_voxels, n_exps)
        Weights matrix from fitted VoxelModel.
    nodes : array-like, shape (n_exps, n_voxels)
        Nodes matrix from fitted VoxelModel.

    Examples
    --------

    >>> from voxel_model.implicit_construction import ImplicitModel
    """
    ndim = 2

    @classmethod
    def from_hdf5(cls, weights_file, nodes_file, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_csv(cls, weights_file, nodes_file, **kwargs):
        """ uses np.loadtxt!!!!!! """
        loader = partial(np.loadtxt, delimiter=",", ndmin=cls.ndim, **kwargs)

        weights, nodes = map(loader, (weights_file, nodes_file))
        return cls(weights, nodes)

    @classmethod
    def from_npy(cls, weights_file, nodes_file, **kwargs):
        """ uses np.load!!!!!! """
        loader = partial(np.load, allow_pickle=True, **kwargs)

        weights, nodes = map(loader, (weights_file, nodes_file))
        return cls(weights, nodes)

    @classmethod
    def from_fitted_voxel_model(cls, voxel_model):
        """ from fitted voxel model """
        try:
            weights = voxel_model.weights
            nodes = voxel_model.y_fit_

        except AttributeError:
            raise ValueError("VoxelModel has not been fit!")

        return cls(weights, nodes)

    def __init__(self, weights, nodes):
        if not ( isinstance(weights, np.ndarray) and
                 isinstance(nodes, np.ndarray ) ):
            raise ValueError( "both weights and nodes must be numpy.ndarray" )

        if weights.shape[1] != nodes.shape[0]:
            raise ValueError( "weights and nodes must have equal "
                              "inner dimension" )

        if weights.dtype != nodes.dtype:
            raise ValueError( "weights and nodes must be of the same dtype" )

        self.weights = weights
        self.nodes = nodes

    def __getitem__(self, key):
        """Allows for slice indexing similar to np.ndarray.

        ...
        """

        if isinstance(key, tuple):
            if len(key) != self.ndim:
                raise ValueError("slice is not compatible with array")

            # row/colum slice
            return self.weights[key[0],:].dot( self.nodes[:,key[1]] )

        else:
            # row : slice, int, list
            return self.weights[key].dot(self.nodes)

    def __len__(self):
        return self.weights.shape[0]

    @property
    def dtype(self):
        return self.weights.dtype #choice doesnt matter

    @property
    def shape(self):
        return (self.weights.shape[0], self.nodes.shape[1])

    @property
    def size(self):
        return reduce(op.mul, self.shape)

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        """DOES NOT RETURN VIEW"""
        self.nodes, self.weights = self.weights.T, self.nodes.T

        return self

    def astype(self, dtype, **kwargs):
        """ ...

        Consistent with numpy.ndarray.astype with copy

        """
        self.weights = self.weights.astype(dtype, **kwargs)
        self.nodes = self.nodes.astype(dtype, **kwargs)

        return self

    def sum(self, axis=None):
        """ .... """
        if axis is None:
            return self.weights.sum(axis=0).dot( self.nodes.sum(axis=1) )

        elif axis == 0:
            return self.weights.sum(axis=axis).dot( self.nodes )

        else:
            # [-1, 1] or IndexError
            return self.weights.dot( self.nodes.sum(axis=axis) )

    def mean(self, axis=None):
        """ ... """

        # IndexError if axis not in [None, -1, 0, 1]
        n = self.size if axis is None else self.shape[axis]

        return self.sum(axis=axis) / n

    def iterrows(self):
        """Generator for yielding rows of the voxel matrix"""
        for row in self.weights:
            yield row.dot(self.nodes)

    def itercolumns(self):
        """Generator for yielding columns of the voxel matrix"""
        for column in self.nodes.T:
            yield self.weights.dot(column)

    def iterrows_blocked(self, n_blocks=0):
        """Generator for yielding rows of the voxel matrix"""

        if n_blocks > self.weights.shape[0] or n_blocks < 1:
            raise ValueError( "invalid number of blocks" )

        row_blocks = np.array_split( self.weights, n_blocks, axis=0 )
        for block in row_blocks:
            yield block.dot( self.nodes )

    def itercolumns_blocked(self, n_blocks=0):
        """Generator for yielding rows of the voxel matrix"""

        if n_blocks > self.weights.shape[0] or n_blocks < 1:
            raise ValueError( "invalid number of blocks" )

        col_blocks = np.array_split( self.nodes, n_blocks, axis=1 )
        for block in col_blocks:
            yield self.weights.dot( block )
