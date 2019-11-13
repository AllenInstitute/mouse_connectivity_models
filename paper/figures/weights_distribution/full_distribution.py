import os
import logging

import allensdk.core.json_utilities as ju
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from mcmodels.core import VoxelModelCache, VoxelData


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(FILE_DIR, '..', '..')

INPUT_JSON = os.path.join(TOP_DIR, 'input.json')
EXPERIMENTS_EXCLUDE_JSON = os.path.join(TOP_DIR, 'experiments_exclude.json')

GREY_ID = 8


def get_model_data(cache, structure_id, eid_set=None, exp_exclude=[], cre=False):
    # NOTE: from figures/model_comparison/helpers/utils.py
    """gets model data from ..."""

    # get experiments
    experiments = cache.get_experiments(injection_structure_ids=[structure_id], cre=cre)
    experiment_ids = [e['id'] for e in experiments]

    # exclude bad, restrict to eid_set
    eid_set = experiment_ids if eid_set is None else eid_set
    experiment_ids = set(experiment_ids) & set(eid_set) - set(exp_exclude)

    # get model data
    data = VoxelData(cache, injection_structure_ids=[structure_id], injection_hemisphere_id=2)
    data.get_experiment_data(experiment_ids)

    return data


def get_nnz_weights(cache, exp_exclude=[]):
    data = get_model_data(cache, GREY_ID, exp_exclude=exp_exclude)
    return data.projections[data.projections.nonzero()].flatten()


def plot_weights(weights, log=True):
    if log:
        weights = np.log10(weights)

    pct = np.percentile(weights, 5)

    f, ax = plt.subplots()
    dp = sns.distplot(weights, ax=ax)

    ax.plot((pct, pct), (0, ax.get_ylim()[1]))
    plt.show()


def main():
    input_data = ju.read(INPUT_JSON)

    # experiments to exclude
    experiments_exclude = ju.read(EXPERIMENTS_EXCLUDE_JSON)

    manifest_file = input_data.get('manifest_file')
    manifest_file = os.path.join(TOP_DIR, manifest_file)

    log_level = input_data.get('log_level', logging.DEBUG)
    logging.getLogger().setLevel(log_level)

    # get cache, metric
    logging.debug("loading experiments")
    cache = VoxelModelCache(manifest_file=manifest_file)
    weights = get_nnz_weights(cache, exp_exclude=experiments_exclude)

    # get all weights
    logging.debug("plotting")
    plot_weights(weights)


if __name__ == "__main__":
    main()
