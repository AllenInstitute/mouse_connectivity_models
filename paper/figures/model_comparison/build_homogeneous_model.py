from __future__ import division
import os
import logging

import numpy as np
import pandas as pd
import allensdk.core.json_utilities as ju

from mcmodels.core import VoxelModelCache
from mcmodels.models import HomogeneousModel
from mcmodels.utils import nonzero_unique

from helpers.model_data import ModelData
from helpers.utils import get_structure_id, get_ordered_summary_structures

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(FILE_DIR, '..', '..')
OUTPUT_DIR = os.path.join(FILE_DIR, 'output')

INPUT_JSON = os.path.join(TOP_DIR, 'input.json')
EXPERIMENTS_EXCLUDE_JSON = os.path.join(TOP_DIR, 'experiments_exclude.json')

ROOT_ID = 997
HIGH_RES = False
THRESHOLD_INJECTION = True

def get_summary_structure_ids(cache):
    structure_tree = cache.get_structure_tree()
    structures = structure_tree.get_structures_by_set_id([687527945])

    return [s['id'] for s in structures if s['id'] not in (934, 1009)]


def fit(cache, eid_set=None, experiments_exclude=[], high_res=False, threshold_injection=True):
    logging.debug('getting data')
    ipsi_data = ModelData(cache, ROOT_ID).get_regional_data(
        eid_set=eid_set, experiments_exclude=experiments_exclude, high_res=high_res,
        threshold_injection=threshold_injection, projection_hemisphere_id=2)

    contra_data = ModelData(cache, ROOT_ID).get_regional_data(
        eid_set=eid_set, experiments_exclude=experiments_exclude, high_res=high_res,
        projection_hemisphere_id=1, threshold_injection=threshold_injection)

    X = ipsi_data.injections
    y = np.hstack((ipsi_data.projections, contra_data.projections))

    logging.debug('fitting')
    reg = HomogeneousModel(kappa=np.inf)
    reg.fit(X, y)

    # get ids
    ss_ids = get_summary_structure_ids(cache)
    injection_key = ipsi_data.injection_mask.get_key(structure_ids=ss_ids, hemisphere_id=2)
    ipsi_key = ipsi_data.projection_mask.get_key(structure_ids=ss_ids, hemisphere_id=2)
    contra_key = contra_data.projection_mask.get_key(structure_ids=ss_ids, hemisphere_id=1)

    injection_regions = nonzero_unique(injection_key)
    ipsi_regions = nonzero_unique(ipsi_key)
    contra_regions = nonzero_unique(contra_key)

    ipsi_w = pd.DataFrame(data=reg.weights[:, :len(ipsi_regions)],
                          index=injection_regions,
                          columns=ipsi_regions)
    contra_w = pd.DataFrame(data=reg.weights[:, len(ipsi_regions):],
                            index=injection_regions,
                            columns=contra_regions)

    return pd.concat((ipsi_w, contra_w), keys=('ipsi', 'contra'), axis=1)


def main():
    input_data = ju.read(INPUT_JSON)

    manifest_file = input_data.get('manifest_file')
    manifest_file = os.path.join(TOP_DIR, manifest_file)

    log_level = input_data.get('log_level', logging.DEBUG)
    logging.getLogger().setLevel(log_level)

    # experiments to exclude
    experiments_exclude = ju.read(EXPERIMENTS_EXCLUDE_JSON)

    # load hyperparameter dict
    suffix = 'high_res' if HIGH_RES else 'standard'

    # get caching object
    cache = VoxelModelCache(manifest_file=manifest_file)

    fit_kwargs = dict(high_res=HIGH_RES, threshold_injection=THRESHOLD_INJECTION,
                      experiments_exclude=experiments_exclude)
    model = fit(cache, **fit_kwargs)

    # write results
    logging.debug('saving')
    output_file = os.path.join(OUTPUT_DIR, 'homogeneous-%s-model.csv' % suffix)
    model.to_csv(output_file)


if __name__ == "__main__":
    main()
