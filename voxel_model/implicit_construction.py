"""
"""
import os
import numpy as np

class ImplicitModel(object):
    """Class for implicit construction of the voxel model from ...
    """

    def __init__(self, dir_path=None, voxel_model=None):
        if dir_path is not None:
            try:
                self.weights = os.path.join(dir_path, "weights.csv")
                self.nodes = os.path.join(dir_path, "nodes.csv")
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
          :: np.array
            row of voxel x voxel connectivity matrix
        """
        return self.weights[i].dot(self.nodes)

    def get_col(self, j):
        """Returns a column of the full voxel connectivity matrix

        Parameters
        ----------
        j :: int
            index of wanted column
        Returns
        -------
          :: np.array
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
          :: np.array
            rows of voxel x voxel connectivity matrix
        """
        return self.weights[row_indices].dot(self.nodes)

    def get_cols(self, col_indices):
        """Returns columns of the full voxel connectivity matrix
        Good for chunked computations on columns

        Parameters
        ----------
        column_indices :: int
            index of wanted column
        Returns
        -------
          :: np.array
            columns of voxel x voxel connectivity matrix
        """
        return self.weights.dot(self.nodes[:,column_indices])

    def iterrows(self):
        """Generator for yielding rows of the voxel matrix"""
        for row in self.weights:
            yield row.dot(self.nodes)

    def itercols(self):
        """Generator for yielding columns of the voxel matrix"""
        for column in self.nodes.T:
            yield self.weights.dot(column)
