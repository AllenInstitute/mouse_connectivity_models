from __future__ import division
from collections import namedtuple
import logging

import numpy as np

from .filters import MaskedFilter, NeighborhoodGraph


class SmoothedModel(object):

    DEFAULT_STRUCTURE_SET_ID = 687527945
    DEFAULT_BOUNDS = ((-2, 2), (-2, 2), (-2, 2))

    @property
    def _default_structure_ids(self):
        structure_tree = self.cache.get_structure_tree()
        structures = structure_tree.get_structures_by_set_id([self.DEFAULT_STRUCTURE_SET_ID])

        return [s['id'] for s in structures if s['id'] not in (934, 1009)]

    def __init__(self, cache, voxel_connectivity_array, source_mask, target_mask,
                 structure_ids=None, filter_type='median', bounds=None):
        self.cache = cache
        self.voxel_connectivity_array = voxel_connectivity_array
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.structure_ids = structure_ids
        self.filter_type = filter_type
        self.bounds = bounds

        if self.structure_ids is None:
            self.structure_ids = self._default_structure_ids

        if self.bounds is None:
            self.bounds = self.DEFAULT_BOUNDS

    @property
    def target_structure_masks(self):
        try:
            return self._target_structure_masks
        except AttributeError:
            TargetStructureMask = namedtuple('TargetStructureMask',
                                             'structure_id, hemisphere_id, mask')
            masks = []
            for sid in self.structure_ids:
                left_mask = self.cache.get_structure_mask(sid)[0]
                right_mask = left_mask.copy()

                left_mask[..., left_mask.shape[-1]//2:] = 0
                right_mask[..., :right_mask.shape[-1]//2] = 0

                masks.append(TargetStructureMask(sid, 1, left_mask))
                masks.append(TargetStructureMask(sid, 2, right_mask))

            self._target_structure_masks = masks
            return self._target_structure_masks

    def filter_projection(self, volume):
        result = np.zeros_like(volume)

        for tsm in self.target_structure_masks:
            logging.debug('smoothing in structure %d, hemisphere %d',
                          tsm.structure_id, tsm.hemisphere_id)

            key = self.target_mask.get_structure_indices(
                structure_ids=[tsm.structure_id], hemisphere_id=tsm.hemisphere_id)

            masked_filter = MaskedFilter(tsm.mask, key=key, bounds=self.bounds)
            masked_filter.filter_volume(volume, kind=self.filter_type, out=result)

        return result

    def fit_structure(self, source_structure_id):
        rows = self.source_mask.get_structure_indices(
            structure_ids=[source_structure_id])

        volume = self.voxel_connectivity_array[rows]
        return self.filter_projection(volume)

    def _get_voxel_structure(self, voxel):
        voxel_structure = self.source_mask.reference_space.annotation[voxel]
        ancestor_ids = self.cache.get_structure_tree().ancestor_ids([voxel_structure])[0]

        for sid in self.structure_ids:
            if sid in ancestor_ids:
                return sid

        raise ValueError('voxel not in self.structure_ids')

    def fit_voxel(self, voxel, source_structure_id=None):
        if source_structure_id is None:
            source_structure_id = self._get_voxel_structure(voxel)

        voxel_row = self.source_mask.get_flattened_voxel_index(voxel)
        source_structure_rows = self.source_mask.get_structure_indices(
            structure_ids=[source_structure_id])

        try:
            idx = source_structure_rows.tolist().index(voxel_row)
        except ValueError:
            # get closest voxel
            # NOTE: could be vastly different, not a proper measure,since idx is flattened
            idx = np.argmin(source_structure_rows - voxel_row)

        volume = self.voxel_connectivity_array[source_structure_rows]
        volume = self.filter_projection(volume)

        return volume[idx].ravel()


class OptimizedSmoothedModel(SmoothedModel):

    def __init__(self, *args, **kwargs):
        super(OptimizedSmoothedModel, self).__init__(*args, **kwargs)
        self.ngraph = NeighborhoodGraph(self.cache, bounds=self.bounds)

    def filter_projection(self, volume):
        return self.ngraph.filter_volume(volume, kind=self.filter_type)
