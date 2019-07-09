"""
Reproduces Table 1: Summary of included data.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: BSD 2

from __future__ import division
import os
import logging
import argparse

import numpy as np
import pandas as pd
from scipy.stats import describe
from scipy.spatial.distance import cdist
import allensdk.core.json_utilities as ju

from mcmodels.core import Mask, VoxelModelCache, VoxelData

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(FILE_DIR, '..', '..')

INPUT_JSON = os.path.join(TOP_DIR, 'input.json')
EXPERIMENTS_EXCLUDE_JSON = os.path.join(TOP_DIR, 'experiments_exclude.json')
OUTPUT_FILE = os.path.join(FILE_DIR, 'summary_statistics.csv')


def get_ss_ids(cache):
    """get ss ids"""
    ss_regions = cache.get_structure_tree().get_structures_by_set_id([687527945])
    return {region["id"] for region in ss_regions if region['id'] not in (934, 1009)}


def get_summary_structures(injection_mask, ss_ids):
    """updates with summary structures"""
    assigned = injection_mask.assigned_structures
    return ss_ids & assigned


def get_centroid_stats(model_data):
    """get stats on minimum distances from voxels to centroids"""
    d = cdist(model_data.injection_mask.coordinates, model_data.centroids)
    # mm
    return describe(0.1*d.min(axis=1))


def get_injection_stats(model_data):
    # mm^3
    return describe(0.001*model_data.injections.sum(axis=1))


def pull_out_centroid_stats(structure_map):
    n_obs, minmax, mean, variance, _, _ = structure_map["centroid_stats"]
    structure_map["area_volume"] = 0.001*n_obs
    structure_map["min_min_d"] = minmax[0]
    structure_map["max_min_d"] = minmax[1]
    structure_map["mean_min_d"] = mean
    structure_map["var_min_d"] = variance


def pull_out_injection_stats(structure_map):
    n_obs, minmax, mean, variance, _, _ = structure_map["injection_stats"]
    structure_map["n_exps"] = n_obs
    structure_map["min_inj_vol"] = minmax[0]
    structure_map["max_inj_vol"] = minmax[1]
    structure_map["mean_inj_vol"] = mean
    structure_map["var_inj_vol"] = variance


def pull_out_region_volume_stats(structure_map):
    region_sizes = [d["region_volume"]
                    for d in structure_map["region_map"].values()]

    n_obs, minmax, mean, variance, _, _ = describe(region_sizes)

    structure_map["min_region_vol"] = minmax[0]
    structure_map["max_region_vol"] = minmax[1]
    structure_map["mean_region_vol"] = mean
    structure_map["var_region_vol"] = variance


def pull_out_region_exps_stats(structure_map):
    counts = [d["n_exps_in"] for d in structure_map["region_map"].values()]

    # lol
    structure_map["r_geq_1"] = sum(i >= 1 for i in counts)
    structure_map["r_geq_2"] = sum(i >= 2 for i in counts)
    structure_map["r_geq_3"] = sum(i >= 3 for i in counts)
    structure_map["r_geq_4"] = sum(i >= 4 for i in counts)
    structure_map["r_geq_5"] = sum(i >= 5 for i in counts)


def get_nexps_regions(cache, model_data, regions, tol=2.0):
    # tol in case centroid is loc inside contained fiber tract

    region_map = {r:dict() for r in regions}
    for region, d in region_map.items():

        # mask to ipsi
        r_mask = cache.get_structure_mask(region)[0]
        r_mask[..., :r_mask.shape[2]//2] = 0

        # get distances
        dists = cdist(np.argwhere(r_mask), model_data.centroids)

        # mm^3
        d["region_volume"] = 0.001*r_mask.sum()
        d["n_exps_in"] = np.any(dists <= tol, axis=0).sum()

    return region_map


def get_full_map(cache, structures, experiments_exclude):
    # get ids
    structure_ids = [s['id'] for s in
                     cache.get_structure_tree().get_structures_by_acronym(structures)]

    # get summary structure ids
    ss_ids = get_ss_ids(cache)
    target_mask = Mask.from_cache(cache, structure_ids=structure_ids, hemisphere_id=3)

    # initialize by pulling data
    full_map = {s : dict() for s in structures}

    for sid, structure_map in zip(structure_ids, full_map.values()):
        # get data
        model_data = VoxelData(
            cache, injection_structure_ids=[sid], injection_hemisphere_id=2,
            flip_experiments=True, normalized_injection=False)

        experiments = cache.get_experiments(cre=False, injection_structure_ids=[sid])
        print(len(experiments))
        eids = set([e['id'] for e in experiments]) - set(experiments_exclude)
        model_data.get_experiment_data(eids)

        regions = get_summary_structures(model_data.injection_mask, ss_ids)
        structure_map.update(dict(regions=regions, n_regions=len(regions)))

        structure_map["region_map"] = get_nexps_regions(cache, model_data, regions)
        structure_map["centroid_stats"] = get_centroid_stats(model_data)
        structure_map["injection_stats"] = get_injection_stats(model_data)

        pull_out_region_volume_stats(structure_map)
        pull_out_region_exps_stats(structure_map)
        pull_out_centroid_stats(structure_map)
        pull_out_injection_stats(structure_map)

    return full_map


def get_df(full_map, structures):

    cols = ["n_exps", "n_regions", "r_geq_1", "r_geq_2",
            "r_geq_3", "r_geq_4", "r_geq_5",
            "area_volume", "min_region_vol", "max_region_vol",
            "mean_region_vol", "var_region_vol", "min_min_d", "max_min_d",
            "mean_min_d", "var_min_d", "min_inj_vol", "max_inj_vol",
            "mean_inj_vol", "var_inj_vol"]

    df =  pd.DataFrame(full_map)
    return df.T.loc[structures, cols]


def main():
    input_data = ju.read(INPUT_JSON)

    structures = input_data.get('structures')

    manifest_file = input_data.get('manifest_file')
    manifest_file = os.path.join(TOP_DIR, manifest_file)

    log_level = input_data.get('log_level', logging.DEBUG)
    logging.getLogger().setLevel(log_level)

    # experiments to exclude
    experiments_exclude = ju.read(EXPERIMENTS_EXCLUDE_JSON)

    # get caching object
    cache = VoxelModelCache(manifest_file=manifest_file)

    # get full map
    full_map = get_full_map(cache, structures, experiments_exclude)

    # convert to df
    df = get_df(full_map, structures)

    # save
    df.to_csv(OUTPUT_FILE)

if __name__ == "__main__":
    main()
