from __future__ import division
from collections import namedtuple
import logging
import time
import abc
import six

import numpy as np

from .py_neighborhood import MaskedNeighborhoodIterator


class _BaseFilter(six.with_metaclass(abc.ABCMeta)):

    @abc.abstractmethod
    def _filter_volume(self, volume, filter_func, out):
        '''helper'''

    def filter_volume(self, volume, kind='median', out=None):
        if volume.ndim != 2:
            raise ValueError('arr must be 2D')

        # filter function
        filter_func = getattr(np, kind)

        if out is None:
            out = np.zeros_like(volume)
        elif out.shape != volume.shape:
            raise ValueError('if supplied, out must be same shape as arr')

        logging.debug('filtering using %s', kind)
        start_time = time.time()
        out = self._filter_volume(volume, filter_func, out)

        logging.debug('time taken: %.3f min', (time.time() - start_time) / 60)
        return out


class MaskedFilter(_BaseFilter):

    def __init__(self, mask, key=None, bounds=None):
        if mask.ndim != 3:
            raise ValueError('mask must be 3D')

        if key is None:
            # assume all rows in mask
            key = np.arange(np.count_nonzero(mask))

        self.mask = mask
        self.key = key
        self.bounds = bounds

        self.mask[mask.nonzero()] = key + 1

    def _filter_volume(self, volume, filter_func, out):
        masked_iter = MaskedNeighborhoodIterator(self.mask, bounds=self.bounds)
        for k, mask_idx in zip(self.key, masked_iter):

            idx = self.mask[mask_idx]
            idx = idx[idx.nonzero()] - 1 # we added 1

            out[:, k] = filter_func(volume.take(idx, axis=1), axis=1)

        return out


_IndexNode = namedtuple('_IndexNode', 'node, edges')
class _IndexGraph(list):

    def add_node(self, node, edges):
        super(_IndexGraph, self).append(_IndexNode(node, edges))


class NeighborhoodGraph(_BaseFilter):

    STRUCTURE_SET_ID = 687527945
    #EXTRA_STRUCTURE_SET_ID = 12
    DEFAULT_BOUNDS = ((-2, 2), (-2, 2), (-2, 2))

    def _get_structure_ids_from_set_id(self, set_id):
        structure_tree = self.cache.get_structure_tree()
        structures = structure_tree.get_structures_by_set_id([set_id])

        return [s['id'] for s in structures if s['id'] not in (934, 1009)]

    @property
    def structure_ids(self):
        try:
            return self._structure_ids
        except AttributeError:
            summary_structures = self._get_structure_ids_from_set_id(self.STRUCTURE_SET_ID)
            #fill_structures = self._get_structure_ids_from_set_id(self.EXTRA_STRUCTURE_SET_ID)
            fill_structures = [315, 698, 1089, 703, 477, 803, 549, 1097, 313, 771, 354, 512]

            # we want filled first
            fill_structures.extend(summary_structures)
            self._structure_ids = fill_structures
            return self._structure_ids

    def __init__(self, cache, bounds=None):
        if bounds is None:
            bounds = self.DEFAULT_BOUNDS

        self.cache = cache
        self.bounds = bounds

    def _get_accum_mask(self):
        logging.debug('computing accumulated mask')
        accum_mask = np.zeros(self.cache.get_annotation_volume()[0].shape)
        counter = range(1, len(self.structure_ids)+1, 2)
        for i, sid in zip(counter, self.structure_ids):
            left_mask = self.cache.get_structure_mask(sid)[0]
            right_mask = left_mask.copy()

            left_mask[..., left_mask.shape[-1]//2:] = 0
            right_mask[..., :right_mask.shape[-1]//2] = 0

            accum_mask[left_mask.nonzero()] = i*left_mask[left_mask.nonzero()]
            accum_mask[right_mask.nonzero()] = (i+1)*right_mask[right_mask.nonzero()]

        return accum_mask

    @property
    def graph(self):
        try:
            return self._graph
        except AttributeError:
            self._graph = _IndexGraph()
            accum_mask = self._get_accum_mask()

            nnz = accum_mask[accum_mask.nonzero()]
            key_mask = np.zeros(accum_mask.shape, dtype=np.uint32)
            key_mask[accum_mask.nonzero()] = np.arange(nnz.size, dtype=np.uint32)

            logging.debug('computing filter graph')
            masked_iter = MaskedNeighborhoodIterator(accum_mask, bounds=self.bounds)
            for curr_val, mask_idx in zip(nnz, masked_iter):
            # TODO: curr_val should be taken from iterator

                # TODO: remove tolist() in mask_idx & reomove np.asarray()
                in_mask = accum_mask[mask_idx] == curr_val
                key_idx = np.asarray(mask_idx)[:, in_mask].tolist()

                node = masked_iter.idx - 1 # NOTE: we added 1????
                edges = key_mask[key_idx]

                if edges.size > 0:
                    self._graph.add_node(node, edges)

            logging.debug('finished computing filter graph')
            return self._graph

    def _filter_volume(self, volume, filter_func, out):
        for i, idx in self.graph:
            out[:, i] = filter_func(volume.take(idx, axis=1), axis=1)

        return out
