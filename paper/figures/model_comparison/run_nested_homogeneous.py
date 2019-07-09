import os
import logging

import allensdk.core.json_utilities as ju
from sklearn.model_selection import LeaveOneOut
from mcmodels.core import VoxelModelCache

from helpers.model_data import ModelData
from helpers.error import HomogeneousModelError
from helpers.utils import get_structure_id, write_output

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

CV = LeaveOneOut()
HIGH_RES = False
THRESHOLD_INJECTION = True


def run_structure(cache, structure_id, cv=None, eid_set=None, experiments_exclude=[],
                  high_res=False, threshold_injection=True):

    data = ModelData(cache, structure_id).get_regional_data(
        eid_set=eid_set, experiments_exclude=experiments_exclude, high_res=high_res,
        threshold_injection=threshold_injection)

    # nested cross val
    logging.debug("Performing nested cross validation: "
                  "(%s samples, %s vars)" % data.projections.shape)
    error = HomogeneousModelError(cache, data, cv=cv)
    error.run()

    return error.scores


def main():
    input_data = ju.read(INPUT_JSON)
    structures = input_data.get('structures')
    manifest_file = input_data.get('manifest_file')
    manifest_file = os.path.join(TOP_DIR, manifest_file)

    log_level = input_data.get('log_level', logging.DEBUG)
    logging.getLogger().setLevel(log_level)

    # experiments to exclude
    experiments_exclude = ju.read(EXPERIMENTS_EXCLUDE_JSON)
    eid_set = ju.read(EXPERIMENTS_PTP_JSON)

    # get caching object
    cache = VoxelModelCache(manifest_file=manifest_file)

    suffix = 'high_res' if HIGH_RES else 'standard'
    output_dir = os.path.join(OUTPUT_DIR, 'homogeneous-%s' % suffix)
    run_kwargs = dict(high_res=HIGH_RES, threshold_injection=THRESHOLD_INJECTION,
                      experiments_exclude=experiments_exclude, cv=CV)

    for structure in structures:
        # get structure id
        logging.debug("Running nested cross validation for structure: %s", structure)
        structure_id = get_structure_id(cache, structure)

        scores = run_structure(cache, structure_id, eid_set=None, **run_kwargs)
        logging.debug("regional score : %.2f", scores['test_regional'].mean())
        write_output(output_dir, structure, structure_id, scores, 'scores_full')

        logging.debug("Scoring only where power to predict")
        try:
            scores = run_structure(cache, structure_id, eid_set=eid_set, **run_kwargs)
            logging.debug("regional score : %.2f", scores['test_regional'].mean())
        except:
            logging.debug("Not enough exps")
        else:
            write_output(output_dir, structure, structure_id, scores, 'scores_ptp')


if __name__ == "__main__":
    main()
