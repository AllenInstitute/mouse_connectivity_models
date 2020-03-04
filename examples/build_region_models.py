'''Script to build the regionalized connectivity (ipsi/contra)'''
import os
import logging
import argparse

import numpy as np
import pandas as pd
from mcmodels.core import VoxelModelCache
from mcmodels.models.voxel import RegionalizedModel

DEFAULT_MANIFEST_FILE = os.path.join(os.path.expanduser('~'), 'connectivity',
                                     'voxel_model_manifest.json')
OUTPUT_DIR = 'model'


def get_ordered_summary_structures(cache):
    """Returns structure ids of summary structures - fiber tracts (and 934)"""
    ss_regions = cache.get_structure_tree().get_structures_by_set_id([687527945])

    # 934 not in 100 micron!!!!! (dont want fiber tracts)
    ids, orders = [], []
    for region in ss_regions:
        if region["id"] not in [934, 1009]:
            ids.append(region["id"])
            orders.append(region["graph_order"])

    # return ids sorted by graph order
    ids = np.asarray(ids)
    return ids[np.argsort(orders)]


def main():
    # set log level
    logging.getLogger().setLevel(args.log_level)

    # initialize cache object
    logging.info('initializing VoxelModelCache with manifest_file: %s',
                 args.manifest_file)
    cache = VoxelModelCache(manifest_file=args.manifest_file)
    structure_ids = get_ordered_summary_structures(cache)

    # load in voxel model
    logging.info('loading array')
    voxel_array, source_mask, target_mask = cache.get_voxel_connectivity_array()

    source_key = source_mask.get_key(structure_ids=structure_ids)
    ipsi_key = target_mask.get_key(structure_ids=structure_ids, hemisphere_id=2)
    contra_key = target_mask.get_key(structure_ids=structure_ids, hemisphere_id=1)

    ipsi_model = RegionalizedModel.from_voxel_array(
        voxel_array, source_key, ipsi_key, ordering=structure_ids, dataframe=True)
    contra_model = RegionalizedModel.from_voxel_array(
        voxel_array, source_key, contra_key, ordering=structure_ids, dataframe=True)

    # get each metric
    get_metric = lambda s: pd.concat((getattr(ipsi_model, s), getattr(contra_model, s)),
                                     keys=('ipsi', 'contra'), axis=1)

    # write results
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # regionalized
    logging.info('saving metrics to directory: %s', OUTPUT_DIR)
    get_metric('connection_density').to_csv(
        os.path.join(OUTPUT_DIR, 'connection_density.csv'))
    get_metric('connection_strength').to_csv(
        os.path.join(OUTPUT_DIR, 'connection_strength.csv'))
    get_metric('normalized_connection_density').to_csv(
        os.path.join(OUTPUT_DIR, 'normalized_connection_density.csv'))
    get_metric('normalized_connection_strength').to_csv(
        os.path.join(OUTPUT_DIR, 'normalized_connection_strength.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_file', default=DEFAULT_MANIFEST_FILE)
    parser.add_argument('--log_level', default=logging.DEBUG)
    args = parser.parse_args()

    main()
