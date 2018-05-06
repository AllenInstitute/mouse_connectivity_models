"""
Module containing :class:`VoxelConnectivityArray`: an object for implicitly
constructing the voxel-scale connectivity matrix on demand.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

from __future__ import division
from functools import partial, reduce
import operator as op

import numpy as np


class VoxelConnectivityArray(object):
    """Class for implicit construction of the voxel model.

    VoxelConnectivityArray is used to perfom analysis on the large (~200,000 by
    400,000 element) voxel-scale connectivity matrix on normal* machines (loading
    the entire matrix would take hundreds of GB of working memory). You can
    access elements of the VoxelConnectivityArray object just like a list or
    numpy array, where each call to __getitem__ will implicitly construct the
    given slice of the array. Additionally, several numpy.ndarray methods have
    been implemented.

    See :class:`VoxelModel` for weights/nodes descriptions.

    Parameters
    ----------
    weights : array-like, shape (n_voxels, n_exps)
        Weights matrix from fitted VoxelModel.

    nodes : array-like, shape (n_exps, n_voxels)
        Nodes matrix from fitted VoxelModel.

    Examples
    --------
    >>> from mcmodels.core import VoxelModelCache
    >>> cache = VoxelModelCache()
    >>> voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()
    >>> # VoxelConnectivityArray has several numpy.ndarray like methods
    >>> # get some arbitrary model weights
    >>> voxel_array[20:22, 10123]
    array([0.000145, 0.000098])
    >>> voxel_array.shape
    (226346, 448962)
    >>> voxel_array.T
    VoxelConnectivityArray(dtype=float32, shape=(448962, 226346))
    """

    ndim = 2

    @classmethod
    def from_csv(cls, weights_file, nodes_file, **kwargs):
        """Alternative constructor loading weights, nodes from `.csv` files.

        Parameters
        ----------
        weights_file : string or path
            Path to the `.csv` file containing the model weights. This file can
            have `.gz` or `.bz2` compression

        nodes_file : string or path
            Path to the `.csv` file containing the model nodes. This file can
            have `.gz` or `.bz2` compression

        **kwargs
            Optional keyword arguments supplied to `numpy.loadtxt
            <https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/
            numpy.loadtxt.html>`_

        Returns
        -------
        An instantiated VoxelConnectivityArray object.
        """
        loader = partial(np.loadtxt, delimiter=",", ndmin=cls.ndim, **kwargs)

        weights, nodes = map(loader, (weights_file, nodes_file))
        return cls(weights, nodes)

    @classmethod
    def from_npy(cls, weights_file, nodes_file, **kwargs):
        """Alternative constructor loading weights, nodes from npy, npz files.

        Parameters
        ----------
        weights_file : string or path
            Path to the `.npy` or `.npz` file containing the model weights.

        nodes_file : string or path
            Path to the `.npy` or `.npz` file containing the model nodes.

        **kwargs
            Optional keyword arguments supplied to `numpy.load
            <https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/
            numpy.load.html>`_

        Returns
        -------
        An instantiated VoxelConnectivityArray object.
        """
        loader = partial(np.load, allow_pickle=True, **kwargs)

        weights, nodes = map(loader, (weights_file, nodes_file))
        return cls(weights, nodes)

    @classmethod
    def from_fitted_voxel_model(cls, voxel_model):
        """Alternative constructor using fitted :class:`VoxelModel` object.

        Parameters
        ----------
        voxel_model : A fitted :class:`VoxelModel` object.

        Returns
        -------
        An instantiated VoxelConnectivityArray object.
        """
        try:
            weights = voxel_model.weights
            nodes = voxel_model.nodes

        except AttributeError:
            raise ValueError("VoxelModel has not been fit!")

        return cls(weights, nodes)

    def __init__(self, weights, nodes):
        # assumes numpy arrays
        if weights.shape[1] != nodes.shape[0]:
            raise ValueError("weights (%d) and nodes (%d) must have equal "
                             "inner dimension" % (weights.shape[1], nodes.shape[0]))

        if weights.dtype != nodes.dtype:
            raise ValueError("weights (%s) and nodes (%s) must be of the same "
                             "dtype" % (weights.dtype, nodes.dtype))

        self.weights = weights
        self.nodes = nodes

    def __repr__(self):
        return '{}(dtype={}, shape={})'.format(
            self.__class__.__name__, self.dtype, self.shape)

    def __getitem__(self, key):
        """Allows for slice indexing similar to np.ndarray."""
        if isinstance(key, tuple):
            if len(key) != self.ndim:
                raise ValueError("slice is not compatible with array")

            # row/colum slice
            return self.weights[key[0], :].dot(self.nodes[:, key[1]])

        else:
            # row : slice, int, list
            return self.weights[key].dot(self.nodes)

    def __len__(self):
        return self.weights.shape[0]

    @property
    def dtype(self):
        """numpy.dtype of full array"""
        # doesn't matter, equivilant dtypes enforced in __init__
        return self.weights.dtype

    @property
    def shape(self):
        """numpy.shape of full array"""
        return (self.weights.shape[0], self.nodes.shape[1])

    @property
    def size(self):
        """numpy.size of full array"""
        return reduce(op.mul, self.shape)

    @property
    def T(self):
        """Short for transpose()"""
        return self.transpose()

    def transpose(self):
        """Returns transpose of full array."""
        self.nodes, self.weights = self.weights.T, self.nodes.T

        return self

    def astype(self, dtype, **kwargs):
        """Consistent with numpy.ndarray.astype.

        see `numpy.ndarray.astype <https://docs.scipy.org/doc/numpy-1.14.0/
        reference/generated/numpy.ndarray.astype.html>`_ for more info.

        Parameters
        ----------
        dtype : string
            Data type to convert array.

        **kwargs :
            Keyword arguments to numpy.ndarray.astype

        Returns
        -------
        self :
            VoxelArray with new dtype.

        """
        self.weights = self.weights.astype(dtype, **kwargs)
        self.nodes = self.nodes.astype(dtype, **kwargs)

        return self

    def sum(self, axis=None):
        """Consistent with numpy.ndarray.sum.

        see `numpy.ndarray.sum <https://docs.scipy.org/doc/numpy-1.14.0/
        reference/generated/numpy.ndarray.sum.html>`_ for more info.

        Parameters
        ----------
        axis - None, int, optional (default=None)
            Axis over which to sum.

        Returns
        -------
        array
            Sum along axis.

        """
        if axis is None:
            return self.weights.sum(axis=0).dot(self.nodes.sum(axis=1))

        elif axis == 0:
            return self.weights.sum(axis=axis).dot(self.nodes)

        # [-1, 1] or IndexError
        return self.weights.dot(self.nodes.sum(axis=axis))

    def mean(self, axis=None):
        """Consistent with numpy.ndarray.mean.

        see `numpy.ndarray.mean <https://docs.scipy.org/doc/numpy-1.14.0/
        reference/generated/numpy.ndarray.mean.html>`_ for more info.

        Parameters
        ----------
        axis - None, int, optional (default=None)
            Axis over which to take mean.

        Returns
        -------
        array
            Mean along axis.

        """
        # IndexError if axis not in [None, -1, 0, 1]
        n = self.size if axis is None else self.shape[axis]

        return self.sum(axis=axis) / n

    def iterrows(self):
        """Generator for yielding rows of the voxel matrix.

        Yields
        ------
        array : shape = (n_columns,)
            Single row of the voxel-scale connectivity matrix.
        """
        for row in self.weights:
            yield row.dot(self.nodes)

    def itercolumns(self):
        """Generator for yielding columns of the voxel matrix.

        Yields
        ------
        array : shape = (n_rows,)
            Single column of the voxel-scale connectivity matrix.
        """
        for column in self.nodes.T:
            yield self.weights.dot(column)

    def iterrows_blocked(self, n_blocks):
        """Generator for yielding blocked rows of the voxel matrix.

        Parameters
        ----------
        n_blocks : int
            The number of blocks of rows that is wished to be returned. Must be
            on the interval [1, n_rows]

        Yields
        ------
        array : A block of rows of the full voxel-scale connectivity matrix.
        """
        if n_blocks > self.weights.shape[0] or n_blocks < 1:
            raise ValueError("invalid number of blocks! n_blocks must be on the "
                             "interval [1, %d], not %d" % (self.weights.shape[0],
                                                           n_blocks))

        row_blocks = np.array_split(self.weights, n_blocks, axis=0)
        for block in row_blocks:
            yield block.dot(self.nodes)

    def itercolumns_blocked(self, n_blocks=0):
        """Generator for yielding blocked columns of the voxel matrix.

        Parameters
        ----------
        n_blocks : int
            The number of blocks of columns that is wished to be returned.
            Must be on the interval [1, n_columns].

        Yields
        ------
        array : A block of columns of the full voxel-scale connectivity matrix.
        """
        if n_blocks > self.nodes.shape[1] or n_blocks < 1:
            raise ValueError("invalid number of blocks! n_blocks must be on the "
                             "interval [1, %d], not %d" % (self.nodes.shape[1],
                                                           n_blocks))

        col_blocks = np.array_split(self.nodes, n_blocks, axis=1)
        for block in col_blocks:
            yield self.weights.dot(block)
