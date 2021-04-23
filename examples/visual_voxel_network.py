'''
Script for extracting ipsilateral connectivity in the visual system.

See the documentation for more examples and API descriptions:
http://mouse-connectivity-models.readthedocs.io/en/latest/
'''
import os
import logging
import numpy as np
from mcmodels.core import VoxelModelCache, Mask

logger = logging.getLogger(name=__name__)

# file path where the data files will be downloaded
MANIFEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'connectivity', 'voxel_model_manifest.json')

# file path where the visual network will be saved (same directory as this file)
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'visual_network.csv')

# see http://atlas.brain-map.org/ for annotation acronyms
VISUAL_AREA_ACRONYMS = ('VISal', 'VISl', 'VISp', 'VISpl', 'VISpm', 'VISli', 'VISpor')


def get_structure_ids_from_acronyms(cache, acronyms):
    structure_tree = cache.get_structure_tree()
    structure_ids = [s['id'] for s
                     in structure_tree.get_structures_by_acronym(acronyms)]

    return structure_ids


def get_voxel_subgraph(cache, structure_ids, hemisphere_id=3):
    ''' hemisphere_ids : { 1 : left (contra), 2 : right (ipsi), 3 : both } '''

    # download/load voxel_array and mask objects
    logger.debug('Loading full voxel-scale network objects')
    voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()

    # get indices of array corresponding to structures (structure_ids must be a list)
    row_idx = source_mask.get_structure_indices(structure_ids=structure_ids,
                                                hemisphere_id=hemisphere_id)
    col_idx = target_mask.get_structure_indices(structure_ids=structure_ids,
                                                hemisphere_id=hemisphere_id)

    logger.debug('Constructing the subgaph as a numpy ndarray')
    return voxel_array[row_idx, col_idx]


def get_subgraph_masks(cache, structure_ids, target_hemisphere_id=3):
    # construct new masks for subgraph
    # useful if further subdivision is wanted or to map back to 3D CCF space
    row_mask = Mask.from_cache(cache, structure_ids=structure_ids, hemisphere_id=2)
    col_mask = Mask.from_cache(
        cache, structure_ids=structure_ids, hemisphere_id=target_hemisphere_id)

    return row_mask, col_mask


def main():
    # get cache object for loading annotation/model objects
    cache = VoxelModelCache(manifest_file=MANIFEST_FILE)

    # get ids for visual areas
    visual_ids = get_structure_ids_from_acronyms(cache, VISUAL_AREA_ACRONYMS)

    # get voxel-scale connectivity of visual network (ipsilateral)
    visual_network = get_voxel_subgraph(cache, visual_ids, hemisphere_id=2)

    # save visual_network
    logger.debug('Saving the visual network to %s', OUTPUT_FILE)
    np.savetxt(OUTPUT_FILE, visual_network, delimiter='')


if __name__ == '__main__':
    # default logging
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    main()
