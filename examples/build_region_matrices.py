# Authors:
# License:

from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from mcmodels.masks import Mask
from mcmodels.regionalized_model import RegionalizedModel
from mcmodels.utils import get_mcc


def get_ordered_summary_structures(mcc):
    # TODO : replace with json of wanted structures

    """Returns structure ids of summary structures - fiber tracts (and 934)"""
    ss_regions = mcc.get_structure_tree().get_structures_by_set_id([167587189])

    # 934 not in 100 micron!!!!! (dont want fiber tracts)
    ids, orders = [], []
    for region in ss_regions:
        if region["id"] not in [934, 1009]:
            ids.append(region["id"])
            orders.append(region["graph_order"])

    # return ids sorted by graph order
    ids = np.asarray(ids)
    return ids[np.argsort(orders)]


def main(manifest_file, data_dir, output_dir, metrics):

    # grab MouseConnectivityCache instance
    mcc = get_mcc(manifest_file)

    # regions over which to unionize
    region_ids = get_ordered_summary_structures(mcc)

    # load data
    try:
        weights = np.load(os.path.join(data_dir, "weights.npy"))
        nodes = np.load(os.path.join(data_dir, "nodes.npy"))
    except FileNotFoundError:
        raise

    # get masks
    source_mask = Mask(mcc, hemisphere=2)
    target_mask = Mask(mcc, hemisphere=3)

    # get annotation key for regions of interest
    source_key = source_mask.get_key(region_ids)
    ipsi_key = target_mask.get_key(region_ids, hemisphere=2)
    contra_key = target_mask.get_key(region_ids, hemisphere=1)

    # instatiate models
    ipsi_model = RegionalizedModel(weights, nodes, source_key,
                                   ipsi_key, ordering=region_ids)
    contra_model = RegionalizedModel(weights, nodes, source_key,
                                     contra_key, ordering=region_ids)

    # build region matrices for each metric
    print("building region matrices")
    for metric in metrics:

        ipsi_matrix = getattr(ipsi_model, metric)
        contra_matrix = getattr(contra_model, metric)

        region_matrix = np.hstack((ipsi_matrix, contra_matrix))

        # save
        filename = os.path.join(output_dir, "{}.npy".format(metric))
        np.save(filename, region_matrix)


if __name__ == "__main__":

    # mcc settings
    MANIFEST_FILE = os.path.join(os.getcwd(), "connectivity",
                                 "mouse_connectivity_manifest.json")

    # i/o settings
    DATA_DIR = os.path.join(os.getcwd(), "data")
    OUTPUT_DIR = os.path.join(os.getcwd(), "output", "region_matrices")

    # wanted regionalized metrics
    METRICS = [
        "connection_strength",
        "connection_density",
        "normalized_connection_strength",
        "normalized_connection_density"
    ]

    # make sure dir exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    main(MANIFEST_FILE, DATA_DIR, OUTPUT_DIR, METRICS)
