# Authors: Joseph Knox josephk@alleninstitute.org
# License: 

import os
import numpy as np

class ImplicitModel(object):
    """Class for implicit construction of the voxel model

    Allows for implicit construction of the voxel model. Contains functions
    percieved to be useful in this end. If additional functionality wanted, 
    please contact author.

    Can be intatiated from:
        * weights & nodes matrices
        * path to directory containing weights & nodes matrices
        * fitted voxel_model.VoxelModel object

    See voxel_model.VoxelModel for weights/nodes descriptions

    Parameters
    ----------
    weights : array-like, optional (default None), shape (n_voxels, n_exps)
        Weights matrix from fitted VoxelModel.
    nodes : array-like, optional (default None), shape (n_exps, n_voxels)
        Nodes matrix from fitted VoxelModel.
    dir_path : string, optional (default None)
        Path to directory containing weights & nodes matrices from previously
        fitted VoxelModel.
    voxel_model : voxel_model.VoxelModel object
        Fitted VoxelModel object.
        MUST BE FITTED!

    Examples
    --------

    >>> from voxel_model.implicit_construction import ImplicitModel
    >>> implicit_model = ImplicitModel(dir_path="./fitted_model_data/")
    >>> implicit_model.get_row(0)
    [0.0000141, 0.0000001, ..., 0.0000093]
    """

    def __init__(self, weights=None, nodes=None, 
                 dir_path=None, voxel_model=None):
        if weights is not None and nodes is not None:
            if weights.shape[1] != nodes.shape[0]:
                raise ValueError("weight and nodes must match in inner diameter")

            self.weights = weights
            self.nodes = nodes
        elif dir_path is not None:
            try:
                weights_path = os.path.join(dir_path, "weights.csv")
                nodes_path = os.path.join(dir_path, "nodes.csv")
                
                self.weights = np.loadtxt(weights_path, delimiter=",")
                self.nodes = np.loadtxt(nodes_path, delimiter=",")

            except IOError:
                raise ValueError("dir_path does not exist")
        elif voxel_model is not None:
            try:
                self.weights = voxel_model.weights
                self.nodes = voxel_model.y_fit_
            except AttributeError:
                raise ValueError("VoxelModel has not been fit!")

    def get_row(self, i):
        """Returns a row of the full voxel connectivity matrix

        Parameters
        ----------
        i :: int
            index of wanted row

        Returns
        -------
        array, shape=(,n_voxels)
            row of voxel x voxel connectivity matrix
        """
        return self.weights[i].dot(self.nodes)

    def get_column(self, j):
        """Returns a column of the full voxel connectivity matrix

        Parameters
        ----------
        j :: int
            index of wanted column

        Returns
        -------
        array, shape=(n_voxels,)
            column of voxel x voxel connectivity matrix
        """
        return self.weights.dot(self.nodes[:,j])

    def get_rows(self, row_indices):
        """Returns rows of the full voxel connectivity matrix

        Good for chunked computations on rows

        Parameters
        ----------
        row_indices :: list of int
            indices of wanted rows

        Returns
        -------
        array, shape=(len(row_indices),n_voxels)
            rows of voxel x voxel connectivity matrix
        """
        return self.weights[row_indices].dot(self.nodes)

    def get_columns(self, column_indices):
        """Returns columns of the full voxel connectivity matrix

        Good for chunked computations on columns

        Parameters
        ----------
        column_indices :: int
            index of wanted column

        Returns
        -------
        array, shape=(n_voxels,len(row_indices))
            columns of voxel x voxel connectivity matrix
        """
        return self.weights.dot(self.nodes[:,column_indices])

    def iterrows(self):
        """Generator for yielding rows of the voxel matrix"""
        for row in self.weights:
            yield row.dot(self.nodes)

    def itercolumns(self):
        """Generator for yielding columns of the voxel matrix"""
        for column in self.nodes.T:
            yield self.weights.dot(column)
