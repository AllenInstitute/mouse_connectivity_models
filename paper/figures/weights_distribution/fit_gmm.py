from __future__ import print_function
import os
import logging

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
import allensdk.core.json_utilities as ju

from mcmodels.core import VoxelModelCache

from helpers.utils import get_cortical_df, get_pt

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(FILE_DIR, '..', '..')

INPUT_JSON = os.path.join(TOP_DIR, 'input.json')
OUTPUT_FILE = os.path.join(FILE_DIR, 'gaussian_mixture_model_fits.csv')

GMM_PARAMS = dict(n_init=1, max_iter=100, tol=1e-2, covariance_type='full')
MAX_COMPONENTS = 10


def fit_gmm(x, max_components=10, **gmm_kwargs):

    models, scores = [], []
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, **gmm_kwargs)
        gmm.fit(x)

        models.append(gmm)
        scores.append(gmm.bic(x))

    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_model = models[best_idx]

    return best_model, best_score


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

    logging.debug("Computing gaussian mixture model fits for max: %s" % MAX_COMPONENTS)

    dfs = (full_ipsi, full_contra, cortex_ipsi, cortex_contra)
    labels = ("full-ipsi", "full-contra", "cortex-ipsi", "cortex-contra")

    frames = []
    for d, l in zip(dfs, labels):
        # log transform
        d = np.log10(d[d > 0]).reshape(-1, 1)

        # normality test
        _, p_value = stats.shapiro(d)

        # gmm
        gmm, bic = fit_gmm(d, MAX_COMPONENTS, **GMM_PARAMS)

        columns = ('mean', 'var', 'weight')
        print("", l, "-"*40, sep="\n")
        print("shapiro-wilk p_value : %.5g" % p_value)
        print("optimal n components : %d" % gmm.n_components)
        print("bic                  : %.5g" % bic)
        print('\t'.join(columns))
        print("----\t---\t------")

        attrs = tuple(map(np.ravel, (gmm.means_, gmm.covariances_, gmm.weights_)))

        for x in zip(*attrs):
            print("%.2f\t%.2f\t%.3f" % x)

        df = pd.DataFrame(dict(zip(columns, attrs)))
        df.index.name = 'n_components'
        frames.append(df)

    df = pd.concat(frames, keys=labels).unstack()
    df.to_csv(OUTPUT_FILE)


if __name__ == "__main__":
    main()
