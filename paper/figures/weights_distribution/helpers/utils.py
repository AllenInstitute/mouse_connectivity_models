from __future__ import division
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist, squareform


def load_metric_matrix(matrix_dir, metric):
    return pd.read_csv(os.path.join(matrix_dir, "%s.csv" % metric),
                       index_col=[0], header=[0, 1])


def get_structure_map(mcc, structures):
    """Get a dictionary mapping from regions to structure ids (ancestors)"""
    tree = mcc.get_structure_tree()
    # NOTE
    aid_map = tree.get_id_acronym_map()
    ida_map = tree.value_map(lambda x: x['id'], lambda y: y['acronym'])

    structure_ids = [aid_map[x] for x in structures]
    d = dict(zip(structure_ids, tree.descendant_ids(structure_ids)))

    return {ida_map[x] : ida_map[k] for k, v in d.items() for x in v}


def get_densities(arr, thresholds):
    return [arr[arr > thresh].size/arr.size for thresh in thresholds]


def get_cortical_df(df, mcc):
    cortex = df.copy()
    structure_map = get_structure_map(mcc, ["Isocortex"])

    rows = pd.Series(df.index.map(structure_map.get))
    cols = pd.Series(df.columns.get_level_values(1).map(structure_map.get))

    return cortex.loc[~rows.isnull().values, ~cols.isnull().values]

def get_pt(data_frames, pt="ipsi", thresh=None):

    def subset(df):
        return df.values[:, df.columns.get_level_values(0) == pt]

    def ravel(df):
        return df.values.ravel()

    def get_valid(subs):
        return np.all(np.stack([x > thresh for x in subs]), axis=0)

    subs = [subset(df) for df in data_frames]

    if thresh is None:
        return [x.ravel() for x in subs]

    # else return valid
    valid = get_valid(subs)
    return [x[valid].ravel() for x in subs]


def get_valid(*args, **kwargs):
    thresh = kwargs.get("thresh", None)
    valid = np.logical_and([x > thresh for x in args])
    if thresh is None:
        return [x.unstack().unstack().values() for x in args]
    return [x[valid].unstack().unstack().values() for x in args]


def get_centroids(ids, cache):

    ipsi, contra = [], []
    for sid in ids:
        ipsi_mask = cache.get_structure_mask(sid)[0]
        contra_mask = ipsi_mask.copy()

        ipsi_mask[..., :ipsi_mask.shape[2]//2] = 0
        contra_mask[..., contra_mask.shape[2]//2:] = 0

        ipsi_crds = np.argwhere(ipsi_mask)
        contra_crds = np.argwhere(contra_mask)

        if len(ipsi_crds):
            ipsi.append(ipsi_crds.mean(axis=0))
        if len(contra_crds):
            contra.append(contra_crds.mean(axis=0))

    return np.vstack(ipsi), np.vstack(contra)


def get_distances(acs, cache, return_mm=True):
    ac_id_map = cache.get_structure_tree().get_id_acronym_map()
    ids = map(ac_id_map.get, acs)

    c_ipsi, c_contra = get_centroids(ids, cache)

    d_ipsi = pdist(c_ipsi)
    d_contra = cdist(c_ipsi, c_contra)

    if return_mm:
        d_ipsi *= 0.1
        d_contra *= 0.1

    return np.hstack([squareform(d_ipsi), d_contra])

def to_dataframe(arr, index, columns):
    df = pd.DataFrame(arr)
    df.index = index
    df.columns = columns

    return df
