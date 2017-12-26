"""
NOTE :: NEED to add testing for __init__ esp for passing fitted voxel_model
"""
from __future__ import division
import pytest
import numpy as np

from voxel_model.implicit_construction import ImplicitModel

@pytest.fixture(scope="module")
def weights():
    return np.array([[5, 3, 2],
                     [7, 2, 1],
                     [1, 4, 5],
                     [2, 6, 2],
                     [3, 4, 3]])

@pytest.fixture(scope="module")
def nodes():
    return np.array([[1, 2, 1, 4, 5, 6, 2, 1],
                     [0, 0, 1, 2, 0, 0, 0, 1],
                     [2, 2, 2, 1, 0, 7, 6, 9]])

@pytest.fixture(scope="module")
def true_voxel_model(weights, nodes):
    return weights.dot(nodes)

@pytest.fixture(scope="module")
def implicit_model(weights, nodes):
    return ImplicitModel(weights=weights, nodes=nodes)

def test_get_row(implicit_model, true_voxel_model):
    n_rows = implicit_model.weights.shape[0]
    for i in range(n_rows):
        np.testing.assert_array_equal(implicit_model.get_row(i), 
                                      true_voxel_model[i])

def test_get_column(implicit_model, true_voxel_model):
    n_cols = implicit_model.nodes.shape[1]
    for j in range(n_cols):
        np.testing.assert_array_equal(implicit_model.get_column(j),
                                      true_voxel_model[:,j])

def test_get_rows(implicit_model, true_voxel_model):
    rows = [0, 2, 3]
    np.testing.assert_array_equal(implicit_model.get_rows(rows),
                                  true_voxel_model[rows])

def test_get_columns(implicit_model, true_voxel_model):
    cols = [0, 2, 3, 6]
    np.testing.assert_array_equal(implicit_model.get_columns(cols),
                                  true_voxel_model[:, cols])

def test_iterrows(implicit_model, true_voxel_model):
    for i, row in enumerate(implicit_model.iterrows()):
        np.testing.assert_array_equal(row, true_voxel_model[i])

def test_itercols(implicit_model, true_voxel_model):
    for j, column in enumerate(implicit_model.itercolumns()):
        np.testing.assert_array_equal(column, true_voxel_model[:, j])
