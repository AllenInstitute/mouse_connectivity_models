from __future__ import division, print_function, absolute_import
from functools import partial
from scipy import linalg
import numpy as np

from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.metrics.regression import _check_reg_targets

def unionize(v, key):
    """unionizes v (:, len(k)) to regions defined in key"""
    v = np.atleast_2d(v)

    if v.shape[1] != key.size:
        raise ValueError("key must be the same size as the n columns in vector!")

    unq = np.unique(key)
    result = np.empty((v.shape[0], unq.size))

    for j, k in enumerate(unq):
        result[:, j] = v[:, np.where(key == k)[0]].sum(axis=1)

    return result


def mean_squared_relative_error(y_true, y_pred, multioutput='uniform_average'):
    """Scorer from ..."""
    _, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, multioutput)
    num = linalg.norm(y_true - y_pred) ** 2
    den = linalg.norm(y_true) ** 2 + linalg.norm(y_pred) ** 2

    return 2 * np.divide(num, den)


def mean_squared_logarithmic_error(y_true, y_pred, epsilon=1, sample_weight=None,
                                   multioutput='uniform_average'):
    """sklearn.metrics.mean_squared_log_error with epsilon"""

    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)

    if not (y_true >= 0).all() and not (y_pred >= 0).all():
        raise ValueError("Mean Squared Logarithmic Error cannot be used when "
                         "targets contain negative values.")

    return mean_squared_error(np.log(y_true + epsilon), np.log(y_pred + epsilon),
                              sample_weight, multioutput)


def regional_mean_squared_error(y_true, y_pred, **kwargs):
    try:
        key = kwargs.pop('key')
    except KeyError:
        raise ValueError("must be called with 'key' kwarg")

    y_true = unionize(y_true, key)
    y_pred = unionize(y_pred, key)

    return mean_squared_error(y_true, y_pred, **kwargs)


def regional_mean_squared_relative_error(y_true, y_pred, **kwargs):
    try:
        key = kwargs.pop('key')
    except KeyError:
        raise ValueError("must be called with 'key' kwarg")

    y_true = unionize(y_true, key)
    y_pred = unionize(y_pred, key)

    return mean_squared_relative_error(y_true, y_pred, **kwargs)


def regional_mean_squared_logarithmic_error(y_true, y_pred, **kwargs):
    try:
        key = kwargs.pop('key')
    except KeyError:
        raise ValueError("must be called with 'key' kwarg")

    y_true = unionize(y_true, key)
    y_pred = unionize(y_pred, key)

    return mean_squared_logarithmic_error(y_true, y_pred, **kwargs)


def mse_rel():
    return make_scorer(mean_squared_relative_error, greater_is_better=False)

def msle(epsilon=1):
    return make_scorer(mean_squared_logarithmic_error, greater_is_better=False, epsilon=epsilon)

def regional_mse(**kwargs):
    return make_scorer(regional_mean_squared_error, greater_is_better=False, **kwargs)

def regional_mse_rel(**kwargs):
    return make_scorer(regional_mean_squared_relative_error, greater_is_better=False, **kwargs)

def regional_msle(**kwargs):
    return make_scorer(regional_mean_squared_logarithmic_error, greater_is_better=False, **kwargs)






def voxelize(v, key, regions):
    """voxelized v (:, len(k)) to regions defined in key"""
    v = np.atleast_2d(v)
    if v.shape[1] != len(regions):
        # NOTE: better error message
        raise ValueError("key is mismatched! (diff size y and key unique)")

    result = np.empty((v.shape[0], key.size))
    for col, k in zip(v.T, regions):
        where = np.where(key == k)[0]
        # normalization
        result[:, where] = np.divide(col, where.size)[:, np.newaxis]

    return result

def oh_voxel_mean_squared_relative_error(y_true, y_pred, **kwargs):
    try:
        key = kwargs.pop('key')
        regions = kwargs.pop('regions')
    except KeyError:
        raise ValueError("must be called with 'key' and 'order' kwargs")

    y_pred = voxelize(y_pred, key, regions)

    if y_pred.shape != y_true.shape:
        raise ValueError("arrays not same shape, key or order must be wrong")

    return mean_squared_relative_error(y_true, y_pred, **kwargs)

def oh_voxel_mse_rel(**kwargs):
    return make_scorer(oh_voxel_mean_squared_relative_error, greater_is_better=False, **kwargs)
