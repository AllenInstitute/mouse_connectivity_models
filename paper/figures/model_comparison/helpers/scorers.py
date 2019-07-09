from __future__ import division
import numpy as np

from sklearn.metrics import make_scorer
from sklearn.metrics.regression import _check_reg_targets

from mcmodels.core import Mask
from mcmodels.utils import squared_norm


class HybridScorer(object):

    DEFAULT_STRUCTURE_SET_ID = 687527945

    @staticmethod
    def voxel_scorer():
        return make_scorer(mean_squared_relative_error, greater_is_better=False)


    @staticmethod
    def regional_scorer(**kwargs):
        return make_scorer(regional_mean_squared_relative_error, greater_is_better=False, **kwargs)

    @property
    def _default_structure_ids(self):
        structure_tree = self.cache.get_structure_tree()
        structures = structure_tree.get_structures_by_set_id([self.DEFAULT_STRUCTURE_SET_ID])

        return [s['id'] for s in structures if s['id'] not in (934, 1009)]

    def __init__(self, cache, structure_ids=None):
        self.cache = cache
        self.structure_ids = structure_ids

        if self.structure_ids is None:
            self.structure_ids = self._default_structure_ids

    @property
    def scoring_dict(self):

        def get_nnz_assigned(key):
            assigned = np.unique(key)
            if assigned[0] == 0:
                return assigned[1:]
            return assigned

        # target is whole brain
        target_mask = Mask.from_cache(cache=self.cache, hemisphere_id=3)

        ipsi_key = target_mask.get_key(structure_ids=self.structure_ids, hemisphere_id=2)
        contra_key = target_mask.get_key(structure_ids=self.structure_ids, hemisphere_id=1)

        reg_kwargs = dict(ipsi_key=ipsi_key,
                          contra_key=contra_key,
                          ipsi_regions=get_nnz_assigned(ipsi_key),
                          contra_regions=get_nnz_assigned(contra_key))

        return dict(voxel=self.voxel_scorer(), regional=self.regional_scorer(**reg_kwargs))

def log_mean_squared_relative_error(y_true, y_pred):
    log = lambda x: np.log10(x + 1e-8)
    return mean_squared_relative_error(log(y_true), log(y_pred))

def log_regional_mean_squared_relative_error(y_true, y_pred, **kwargs):
    log = lambda x: np.log10(x + 1e-8)
    return regional_mean_squared_relative_error(log(y_true), log(y_pred), **kwargs)


class LogHybridScorer(HybridScorer):

    @staticmethod
    def voxel_scorer():
        return make_scorer(log_mean_squared_relative_error, greater_is_better=False)

    @staticmethod
    def regional_scorer(**kwargs):
        return make_scorer(log_regional_mean_squared_relative_error,
                           greater_is_better=False, **kwargs)


def unionize(v, ipsi_key, contra_key, ipsi_regions, contra_regions):
    """unionizes v (:, len(k)) to regions defined in key"""
    if ipsi_key.shape != contra_key.shape:
        # NOTE: better error message
        raise ValueError("keys are incompatible")

    v = np.atleast_2d(v)
    if v.shape[1] != ipsi_key.size: # or contra, doesnt matter
        raise ValueError("key must be the same size as the n columns in vector!")

    j = 0
    result = np.empty((v.shape[0], len(ipsi_regions) + len(contra_regions)))
    for key, regions in zip((ipsi_key, contra_key), (ipsi_regions, contra_regions)):
        for k in regions:
            result[:, j] = v[:, np.where(key == k)[0]].sum(axis=1)
            j += 1

    return result


def mean_squared_relative_error(y_true, y_pred, multioutput='uniform_average'):
    """Scorer from ..."""
    _, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, multioutput)
    #result = squared_norm(y_true - y_pred) / max(squared_norm(y_true), squared_norm(y_pred))
    result = 2 * squared_norm(y_true - y_pred) / (squared_norm(y_true) + squared_norm(y_pred))
    return result


def regional_mean_squared_relative_error(y_true, y_pred, **kwargs):
    try:
        ipsi_key = kwargs.pop('ipsi_key')
        contra_key = kwargs.pop('contra_key')
        ipsi_regions = kwargs.pop('ipsi_regions')
        contra_regions = kwargs.pop('contra_regions')
    except KeyError:
        raise ValueError("must be called with 'key' and 'order' kwargs")

    y_true = unionize(y_true, ipsi_key, contra_key, ipsi_regions, contra_regions)
    y_pred = unionize(y_pred, ipsi_key, contra_key, ipsi_regions, contra_regions)

    return mean_squared_relative_error(y_true, y_pred, **kwargs)


def mse_rel():
    return make_scorer(mean_squared_relative_error, greater_is_better=False)


def regional_mse_rel(**kwargs):
    return make_scorer(regional_mean_squared_relative_error, greater_is_better=False, **kwargs)
