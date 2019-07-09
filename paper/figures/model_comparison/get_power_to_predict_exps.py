from __future__ import division, print_function, absolute_import
import os
import argparse

import allensdk.core.json_utilities as ju

from mcmodels.core import VoxelModelCache, Mask

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(FILE_DIR, '..', '..')

INPUT_JSON = os.path.join(TOP_DIR, 'input.json')
OUTPUT_FILE = os.path.join(TOP_DIR, 'experiments_ptp.json')


def get_ss_ids(cache):
    """get ss ids"""
    ss_regions = cache.get_structure_tree().get_structures_by_set_id([687527945])
    return {region["id"] for region in ss_regions if region['id'] not in (934, 1009)}


def get_summary_structures(mask, ss_ids):
    """updates with summary structures"""
    assigned = mask.assigned_structures
    return ss_ids & assigned


def get_them(cache, n_min=1, cre=False):

    mask = Mask.from_cache(cache, hemisphere_id=3)
    regions = get_summary_structures(mask, get_ss_ids(cache))

    experiment_ids = []
    for rid in regions:
        exps = cache.get_experiments(injection_structure_ids=[rid], cre=False)
        eids = [e['id'] for e in exps]

        if len(eids) >= n_min:
            experiment_ids.extend(eids)

    return experiment_ids


def main(args):
    input_data = ju.read(INPUT_JSON)
    manifest_file = input_data.get('manifest_file')
    manifest_file = os.path.join(TOP_DIR, manifest_file)

    # get cache, metric
    cache = VoxelModelCache(manifest_file=manifest_file)

    # get ptp experiment ids
    exp_ids = get_them(cache, args.n_min)

    # write
    ju.write(OUTPUT_FILE, exp_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('n_min', type=int)
    args = parser.parse_args()

    main(args)
