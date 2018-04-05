import pytest
import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal, assert_raises, assert_allclose

from mcmodels.models.voxel import RegionalizedModel


# ============================================================================
# RegionalizedModel
# ============================================================================
def test_init():
    # ------------------------------------------------------------------------
    # tests incompatible source key sizes
    weights = np.ones((10, 5))
    nodes = np.ones((5, 10))
    source_key = np.arange(9)
    target_key = np.arange(10)

    assert_raises(ValueError, RegionalizedModel, weights, nodes,
                  source_key, target_key)

    # ------------------------------------------------------------------------
    # tests incompatible target key sizes
    source_key = np.arange(10)
    target_key = np.arange(9)

    assert_raises(ValueError, RegionalizedModel, weights, nodes,
                  source_key, target_key)


def test_predict():
    # ------------------------------------------------------------------------
    # tests notimplemented error
    X = np.ones((5, 10))

    assert_raises(NotImplementedError, RegionalizedModel.predict, None, X)


def test_regionalize_voxel_connectivity_array():
    # ------------------------------------------------------------------------
    # test output is correct
    weights = np.ones((10, 5))
    nodes = np.ones((5, 20))

    # 1:3, 2:1, 3:1, 4:2, 5:2
    source_key = np.array([1, 5, 5, 3, 4, 4, 0, 1, 1, 2])

    # 1:3, 2:2, 3:2, 4:2, 5:3, 6:4
    target_key = np.array([0, 0, 5, 5, 5, 1, 2, 3, 4, 0,
                           0, 1, 1, 2, 3, 4, 6, 6, 6, 6])

    # simple, take n_rows * n_cols * inner_dim (5 in this case)
    true_matrix = np.array([[45, 30, 30, 30, 45, 60],
                            [15, 10, 10, 10, 15, 20],
                            [15, 10, 10, 10, 15, 20],
                            [30, 20, 20, 20, 30, 40],
                            [30, 20, 20, 20, 30, 40]])

    model = RegionalizedModel(weights, nodes, source_key, target_key)
    matrix = model._regionalize_voxel_connectivity_array()

    assert_array_equal(matrix, true_matrix)


def test_get_region_matrix():
    # ------------------------------------------------------------------------
    # tests ordering
    weights = np.ones((10, 5))
    nodes = np.ones((5, 20))
    ordering = [1, 3, 2, 4, 6, 5]

    # 1:3, 2:1, 3:1, 4:2, 5:2
    source_key = np.array([1, 5, 5, 3, 4, 4, 0, 1, 1, 2])

    # 1:3, 2:2, 3:2, 4:2, 5:3, 6:4
    target_key = np.array([0, 0, 5, 5, 5, 1, 2, 3, 4, 0,
                           0, 1, 1, 2, 3, 4, 6, 6, 6, 6])

    # simple, take n_rows * n_cols * inner_dim (5 in this case)
    true_matrix = np.array([[45, 30, 30, 30, 60, 45],
                            [15, 10, 10, 10, 20, 15],
                            [15, 10, 10, 10, 20, 15],
                            [30, 20, 20, 20, 40, 30],
                            [30, 20, 20, 20, 40, 30]])

    model = RegionalizedModel(weights, nodes, source_key, target_key,
                              ordering=ordering)
    matrix = model._get_region_matrix()

    assert_array_equal(matrix, true_matrix)

    # ------------------------------------------------------------------------
    # tests dataframe
    model = RegionalizedModel(weights, nodes, source_key, target_key,
                              ordering=ordering, dataframe=True)
    matrix = model._get_region_matrix()

    assert isinstance(matrix, pd.DataFrame)
    assert_array_equal(matrix, true_matrix)
    assert_allclose(matrix.index, [1, 3, 2, 4, 5])
    assert_allclose(matrix.columns, ordering)


def test_connection_density():
    # ------------------------------------------------------------------------
    # test output is correct
    weights = np.ones((10, 5))
    nodes = np.ones((5, 20))
    source_key = np.array([1, 5, 5, 3, 4, 4, 0, 1, 1, 2])

    # 1:3, 2:2, 3:2, 4:2, 5:3, 6:4
    target_key = np.array([0, 0, 5, 5, 5, 1, 2, 3, 4, 0,
                           0, 1, 1, 2, 3, 4, 6, 6, 6, 6])

    # simple, take n_cols * inner_dim (5 in this case)
    true_matrix = np.array([[15, 10, 10, 10, 15, 20],
                            [15, 10, 10, 10, 15, 20],
                            [15, 10, 10, 10, 15, 20],
                            [15, 10, 10, 10, 15, 20],
                            [15, 10, 10, 10, 15, 20]])

    model = RegionalizedModel(weights, nodes, source_key, target_key)
    matrix = model.normalized_connection_strength

    assert_array_equal(matrix, true_matrix)

    # ------------------------------------------------------------------------
    # test dataframe produces same result
    model = RegionalizedModel(weights, nodes, source_key, target_key,
                              dataframe=True)
    matrix = model.normalized_connection_strength

    assert_array_equal(matrix, true_matrix)


def test_normalized_connection_strength():
    # ------------------------------------------------------------------------
    # test output is correct
    weights = np.ones((10, 5))
    nodes = np.ones((5, 20))

    # 1:3, 2:1, 3:1, 4:2, 5:2
    source_key = np.array([1, 5, 5, 3, 4, 4, 0, 1, 1, 2])
    target_key = np.array([0, 0, 5, 5, 5, 1, 2, 3, 4, 0,
                           0, 1, 1, 2, 3, 4, 6, 6, 6, 6])

    # simple, take n_rows * inner_dim (5 in this case)
    true_matrix = np.array([[15, 15, 15, 15, 15, 15],
                            [5,  5,   5,  5,  5,  5],
                            [5,  5,   5,  5,  5,  5],
                            [10, 10, 10, 10, 10, 10],
                            [10, 10, 10, 10, 10, 10]])

    model = RegionalizedModel(weights, nodes, source_key, target_key)
    matrix = model.connection_density

    assert_array_equal(matrix, true_matrix)

    # ------------------------------------------------------------------------
    # test dataframe produces same result
    model = RegionalizedModel(weights, nodes, source_key, target_key,
                              dataframe=True)
    matrix = model.connection_density

    assert_array_equal(matrix, true_matrix)


def test_normalized_connection_density():
    # ------------------------------------------------------------------------
    # test output is correct
    weights = np.ones((10, 5))
    nodes = np.ones((5, 20))
    source_key = np.array([1, 5, 5, 3, 4, 4, 0, 1, 1, 2])
    target_key = np.array([0, 0, 5, 5, 5, 1, 2, 3, 4, 0,
                           0, 1, 1, 2, 3, 4, 6, 6, 6, 6])

    # simple, just inner dim (5 in this case)
    true_matrix = 5*np.ones((5, 6))


    model = RegionalizedModel(weights, nodes, source_key, target_key)
    matrix = model.normalized_connection_density

    assert_array_equal(matrix, true_matrix)

    # ------------------------------------------------------------------------
    # test dataframe produces same result
    model = RegionalizedModel(weights, nodes, source_key, target_key,
                              dataframe=True)
    matrix = model.normalized_connection_density

    assert_array_equal(matrix, true_matrix)
