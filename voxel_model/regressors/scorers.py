"""Scorers based around mean squared relative error

...

    MSE_rel =
"""
from scipy import linalg
import numpy as np

from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.metrics.regression import _check_reg_targets

from ..utils import nonzero_unique


def _unionize(v, ipsi_key, contra_key):
    """unionizes v (:, len(k)) to regions defined in key"""
    ipsi_key = np.asarray(ipsi_key)
    contra_key = np.asarray(contra_key)

    if ipsi_key.shape != contra_key.shape:
        # NOTE: better error message
        raise ValueError("keys are incompatible")

    v = np.atleast_2d(v)
    if v.shape[1] != ipsi_key.size: # or contra, doesnt matter
        raise ValueError("key must be the same size as the n columns in vector!")

    # get regions
    ipsi_regions = nonzero_unique(ipsi_key)
    contra_regions = nonzero_unique(contra_key)

    j = 0
    result = np.empty((v.shape[0], len(ipsi_regions) + len(contra_regions)))
    for key, regions in zip((ipsi_key, contra_key), (ipsi_regions, contra_regions)):
        for k in regions:
            result[:, j] = v[:, np.where(key == k)[0]].sum(axis=1)
            j += 1

    return result


def _voxelize(v, ipsi_key, contra_key):
    """voxelized v (:, len(k)) to regions defined in key"""
    ipsi_key = np.asarray(ipsi_key)
    contra_key = np.asarray(contra_key)

    if ipsi_key.shape != contra_key.shape:
        # NOTE: better error message
        raise ValueError("keys are incompatible")

    # get regions
    ipsi_regions = nonzero_unique(ipsi_key)
    contra_regions = nonzero_unique(contra_key)

    v = np.atleast_2d(v)
    if v.shape[1] != len(ipsi_regions)+len(contra_regions):
        # NOTE: better error message
        raise ValueError("key is mismatched! (diff size y and key unique)")

    result = np.empty((v.shape[0], ipsi_key.size)) # or contra, the same
    for key, regions in zip((ipsi_key, contra_key), (ipsi_regions, contra_regions)):
        for col, k in zip(v.T, regions):
            where = np.where(key == k)[0]
            result[:, where] = np.divide(col, where.size)[:, np.newaxis]

    return result


def mean_squared_relative_error(y_true, y_pred, multioutput='uniform_average'):
    """Scorer from ..."""
    _, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, multioutput)
    num = linalg.norm(y_true - y_pred) ** 2
    den = linalg.norm(y_true) ** 2 + linalg.norm(y_pred) ** 2

    return 2 * np.divide(num, den)


def regional_mean_squared_relative_error(y_true, y_pred, **kwargs):
    """regionalized mse_rel (from voxel)"""
    try:
        ipsi_key = kwargs.pop('ipsi_key')
        contra_key = kwargs.pop('contra_key')
    except KeyError:
        raise ValueError("must be called with 'ipsi_key' and 'contra_key' kwargs")

    y_true = _unionize(y_true, ipsi_key, contra_key)
    y_pred = _unionize(y_pred, ipsi_key, contra_key)

    return mean_squared_relative_error(y_true, y_pred, **kwargs)


def voxel_mean_squared_relative_error(y_true, y_pred, **kwargs):
    """voxelized mse_rel (from regional)

    ASSUMES y_pred is regional and y_true is voxel
    """
    try:
        ipsi_key = kwargs.pop('ipsi_key')
        contra_key = kwargs.pop('contra_key')
    except KeyError:
        raise ValueError("must be called with 'ipsi_key' and 'contra_key' kwargs")

    y_pred = _voxelize(y_pred, ipsi_key, contra_key)

    if y_pred.shape != y_true.shape:
        raise ValueError("arrays not same shape, key or order must be wrong")

    return mean_squared_relative_error(y_true, y_pred, **kwargs)


def mse_rel():
    return make_scorer(mean_squared_relative_error, greater_is_better=False)


def regional_mse_rel(**kwargs):
    return make_scorer(regional_mean_squared_relative_error, greater_is_better=False, **kwargs)


def voxel_mse_rel(**kwargs):
    return make_scorer(voxel_mean_squared_relative_error, greater_is_better=False, **kwargs)
