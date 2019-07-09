import os
import logging

import pandas as pd
import allensdk.core.json_utilities as ju

from mcmodels.core import VoxelModelCache

from helpers.functions import LogLog, LogLinear
from helpers.ltsq_models import LtsqFit
from helpers.utils import get_cortical_df, get_pt, get_distances, to_dataframe

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(FILE_DIR, '..', '..')

INPUT_JSON = os.path.join(TOP_DIR, 'input.json')
OUTPUT_FILE = os.path.join(FILE_DIR, 'powerlaw_exponential_fits.csv')

def results_to_df(fit):
    KEYS = ['rmse']
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

    # region acs
    region_acs = df_metric.index.values

    logging.debug("computing distances")
    d = get_distances(region_acs, cache)
    d = to_dataframe(d, df_metric.index, df_metric.columns)
    d_cortex = get_cortical_df(d, cache)

    # get projection types
    full_ipsi = get_pt((d, df_metric), thresh=0)
    full_contra = get_pt((d, df_metric), thresh=0, pt="contra")
    cortex_ipsi = get_pt((d_cortex, df_cortex), thresh=0)
    cortex_contra = get_pt((d_cortex, df_cortex), thresh=0, pt="contra")

    # set up
    funcs = [LogLog(), LogLinear()]
    fitter = LtsqFit(funcs)

    dws = (full_ipsi, full_contra, cortex_ipsi, cortex_contra)
    labels = ("full-ipsi", "full-contra", "cortex-ipsi", "cortex-contra")

    frames = []
    logging.debug("computing fits")
    for (d, w), l in zip(dws, labels):
        fitter.fit(d, w)
        logging.debug(l)
        logging.debug(fitter)

        frames.append(results_to_df(fitter))

    df = pd.concat(frames, keys=labels).unstack()
    df.to_csv(OUTPUT_FILE)


if __name__ == "__main__":
    main()
