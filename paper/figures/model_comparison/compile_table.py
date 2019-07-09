from itertools import product
import os
import logging
import argparse

import numpy as np
import pandas as pd
import allensdk.core.json_utilities as ju

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.join(FILE_DIR, '..', '..')
INPUT_JSON = os.path.join(TOP_DIR, 'input.json')
OUTPUT_DIR = os.path.join(FILE_DIR, 'output')

SETS = ["full", "ptp"]
KEYS = ["train_voxel", "train_regional", "test_voxel", "test_regional"]
COL_ORDER = [('full', 'test_voxel'), ('full', 'train_voxel'), ('full', 'test_regional'),
             ('full', 'train_regional'), ('ptp', 'test_regional'), ('ptp', 'train_regional')]


def load_dicts(structures, runs, keys, outdir):
    d = {s:dict(full=dict(), ptp=dict()) for s in structures}
    for structure in structures:
        full = ju.read(os.path.join(outdir, "%s_scores_full.json" % structure))
        ptp = ju.read(os.path.join(outdir, "%s_scores_ptp.json" % structure))

        d[structure]['full'] = {k:-np.mean(full.get(k, [np.nan])) for k in keys}
        d[structure]['ptp'] = {k:-np.mean(ptp.get(k, [np.nan])) for k in keys}

    return d


def get_table(d, structures, runs, keys):
    # reshape for table
    df = dict()
    for s, d in d.items():
        df[s] = {(r, k) : d[r][k] for r, k in product(runs, keys)}
    df = pd.DataFrame(df).T

    return df.loc[structures, list(product(runs, keys))]


def get_df(structures, runs, keys, results_dir):
    d = load_dicts(structures, runs, keys, results_dir)
    df = get_table(d, structures, runs, keys)

    return df


def main(runs):

    input_data = ju.read(INPUT_JSON)
    structures = input_data.get('structures')
    manifest_file = input_data.get('manifest_file')
    manifest_file = os.path.join(TOP_DIR, manifest_file)

    log_level = input_data.get('log_level', logging.DEBUG)
    logging.getLogger().setLevel(log_level)

    # get table for each run
    tables = [get_df(structures, SETS, KEYS, os.path.join(OUTPUT_DIR, run))
              for run in runs]

    # concat tables
    df = pd.concat(tables, keys=runs)

    # reorder
    rows = list(product(structures, runs))
    df = df.swaplevel().loc[rows, COL_ORDER]

    # save
    output_file = os.path.join(OUTPUT_DIR, 'cv_results_%s.csv' % '_'.join(runs))
    df.to_csv(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compile cv results')
    parser.add_argument('runs', metavar='RUNx', type=str, nargs='+',
                        help='a directory containing cv results for a run')
    args = parser.parse_args()

    main(args.runs)
