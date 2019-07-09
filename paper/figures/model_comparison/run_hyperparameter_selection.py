from math import sqrt
import os
import logging

import numpy as np
import allensdk.core.json_utilities as ju
from mcmodels.core import VoxelModelCache

from helpers.model_data import ModelData
from helpers.error import VoxelModelError
from helpers.utils import get_structure_id

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(FILE_DIR, '..', '..')
OUTPUT_DIR = os.path.join(FILE_DIR, 'output')

INPUT_JSON = os.path.join(TOP_DIR, 'input.json')
EXPERIMENTS_EXCLUDE_JSON = os.path.join(TOP_DIR, 'experiments_exclude.json')
EXPERIMENTS_PTP_JSON = os.path.join(TOP_DIR, 'experiments_ptp.json')
if not os.path.exists(EXPERIMENTS_PTP_JSON):
    raise RuntimeError('file %s does not yet exist or has been deleted. '
                       'please run "python get_power_to_predict.py 2" from '
                       'inside the directory %s' % (EXPERIMENTS_PTP_JSON, FILE_DIR))

KERNEL = 'rbf' # rbf, polynomial
OPTION = 'standard' # standard, log, injection


def fit_structure(cache, structure_id, experiments_exclude, kernel=None,
                  model_option='standard'):
    data = ModelData(cache, structure_id).get_voxel_data(
        experiments_exclude=experiments_exclude)

    # nested cross val
    logging.debug("Performing cross validation: (%d samples, %d vars)",
                  *data.projections.shape)
    error = VoxelModelError(cache, data, kernel=kernel)
    reg = error.single_cv(option=model_option)

    logging.debug("score          : %.2f", reg.best_score_)
    if kernel == 'polynomial':
        results = dict(shape=reg.kernel.shape, support=reg.kernel.support)
        logging.debug("optimal shape  : %.0f", results['shape'])
        logging.debug("optimal support: %.0f", results['support'])
    else:
        results = dict(gamma=reg.gamma)
        logging.debug("optimal gamma  : %.3f", results['gamma'])
        logging.debug("(optimal sigma : %.3f)", 1 / sqrt(results['gamma']))

    return results


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

    output_file = os.path.join(OUTPUT_DIR, 'hyperparameters-%s.json' % OPTION)

    results = dict()
    for structure in structures:
        logging.debug("Running cross validation for structure: %s", structure)
        structure_id = get_structure_id(cache, structure)

        results[structure] = fit_structure(cache, structure_id, experiments_exclude,
                                           kernel=KERNEL, model_option=OPTION)

    # write results
    ju.write(output_file, results)


if __name__ == "__main__":
    main()
