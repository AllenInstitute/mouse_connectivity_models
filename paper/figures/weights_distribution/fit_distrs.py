from __future__ import print_function
import os
import logging

import numpy as np
import pandas as pd
import scipy.stats as stats

import allensdk.core.json_utilities as ju

from mcmodels.core import VoxelModelCache

from helpers.model_selection import DistFit
from helpers.utils import get_cortical_df, get_pt

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(FILE_DIR, '..', '..')

DISTRIBUTIONS = ["lognorm", "invgauss", "expon", "norm"]
INPUT_JSON = os.path.join(TOP_DIR, 'input.json')
OUTPUT_FILE = os.path.join(FILE_DIR, 'distance_dependence_fits.csv')


def results_to_df(fit):
    KEYS = ['bic', 'p_value']
    d = fit.to_dict()
    df = pd.DataFrame(d)

    return df.loc[KEYS]

def main():
    input_data = ju.read(INPUT_JSON)

    manifest_file = input_data.get('manifest_file')
    manifest_file = os.path.join(TOP_DIR, manifest_file)

    log_level = input_data.get('log_level', logging.DEBUG)
    logging.getLogger().setLevel(log_level)

    # get cache, metric
    logging.debug("loading regional matrix")
    cache = VoxelModelCache(manifest_file=manifest_file)
    df_metric = cache.get_normalized_connection_density(dataframe=True)


    logging.debug("getting cortical network")
    df_cortex = get_cortical_df(df_metric, cache)

    # get projection types
    full_ipsi, cortex_ipsi = get_pt((df_metric, df_cortex))
    full_contra, cortex_contra = get_pt((df_metric, df_cortex), pt="contra")

    logging.debug("Computing distribution fits for")
    logging.debug("%s" % DISTRIBUTIONS)

    fitter = DistFit(DISTRIBUTIONS)

    dfs = (full_ipsi, full_contra, cortex_ipsi, cortex_contra)
    labels = ("full-ipsi", "full-contra", "cortex-ipsi", "cortex-contra")

    frames = []
    for d, l in zip(dfs, labels):
        fitter.fit(d[d>0])
        logging.debug(l)
        logging.debug(str(fitter))
        frames.append(results_to_df(fitter))

    df = pd.concat(frames, keys=labels).unstack()
    df.to_csv(OUTPUT_FILE)


if __name__ == "__main__":
    main()
