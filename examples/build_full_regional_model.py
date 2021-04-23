'''Script to build the regionalized connectivity in every region annotated at 100 micron.

NOTE: This script takes quite a while to run. If you are simply interested in
      the connectivity within the set summary structures (or any disjoint
      structure set), using the RegionalizedModel class will be much more
      efficient.
'''
import os
import itertools

import numpy as np
import pandas as pd
from mcmodels.core import VoxelModelCache

MANIFEST_FILE = os.path.join(os.path.expanduser('~'), 'connectivity',
                             'voxel_model_manifest.json')
MAJOR_BRAIN_SET_ID = 687527670
OUTPUT_DIR = 'model'


def main():
    # initialize cache object
    cache = VoxelModelCache(manifest_file=MANIFEST_FILE)
    rs = cache.get_reference_space()
    rs.remove_unassigned(update_self=True)

    # major id children only
    structures = []
    for s in rs.structure_tree.get_structures_by_set_id([MAJOR_BRAIN_SET_ID]):
        structures.extend(rs.structure_tree.descendants([s['id']])[0])

    # load in voxel model
    print('loading array')
    voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()

    print('getting keys')
    source_keys, target_keys = [], []
    source_counts, target_counts = [], []
    for s in structures:
        s_mask = source_mask.reference_space.make_structure_mask([s['id']], direct_only=False)
        t_mask = target_mask.reference_space.make_structure_mask([s['id']], direct_only=False)

        # NOTE: ipsi
        t_mask[..., :t_mask.shape[-1]//2] = 0

        # keys
        s_key = source_mask.mask_volume(s_mask).nonzero()[0]
        t_key = target_mask.mask_volume(t_mask).nonzero()[0]

        source_keys.append(s_key)
        target_keys.append(t_key)
        source_counts.append(s_key.size)
        target_counts.append(t_key.size)

    del source_mask
    del target_mask

    # arrays
    source_counts = np.asarray(source_counts)
    target_counts = np.asarray(target_counts)

    # compute
    print('computing regional')
    connection_strength = np.empty(2*[len(structures)])
    for i, j in itertools.product(range(len(structures)), repeat=2):
        print(i, j)
        connection_strength[i, j] = voxel_array[source_keys[i], target_keys[j]].sum()

    del voxel_array
    del source_keys
    del target_keys

    structure_acronyms = [s['acronym'] for s in structures]
    connection_strength = pd.DataFrame(
        connection_strength, index=structure_acronyms, columns=structure_acronyms)

    # other metrics
    connection_density = np.divide(connection_strength, source_counts[:, np.newaxis])
    normalized_connection_strength = np.divide(connection_strength, target_counts[:, np.newaxis])
    normalized_connection_density = np.divide(
        connection_strength, np.outer(source_counts, target_counts))

    # save
    connection_strength.to_csv(os.path.join(OUTPUT_DIR, 'connection_strength.csv'))
    connection_density.to_csv(os.path.join(OUTPUT_DIR, 'connection_density.csv'))
    normalized_connection_strength.to_csv(
        os.path.join(OUTPUT_DIR, 'normalized_connection_strength.csv'))
    normalized_connection_density.to_csv(
        os.path.join(OUTPUT_DIR, 'normalized_connection_density.csv'))


if __name__ == '__main__':
    main()
