import os

import numpy as np
import allensdk.core.json_utilities as ju
from sklearn.model_selection import cross_validate, GridSearchCV

from mcmodels.core import VoxelData


def get_structure_id(cache, acronym):
    try:
        return cache.get_structure_tree().get_structures_by_acronym([acronym])[0]['id']
    except KeyError:
        raise ValueError("structure acronym (%s) is not valid" % acronym)


def get_model_data(cache, structure_id, eid_set=None, exp_exclude=[], cre=False):
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


def write_output(output_dir, acronym, sid, data_dict, suffix=''):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ouput_json = os.path.join(output_dir, "%s_%s.json" % (acronym, suffix))

    output_data = dict(structure_acronym=acronym, structure_id=sid)
    output_data.update(data_dict)

    ju.write(ouput_json, output_data)


def nonzero_min(arr):
    return np.min(arr[arr.nonzero()])


def get_ordered_summary_structures(mcc):
    # TODO : replace with json of wanted structures

    """Returns structure ids of summary structures - fiber tracts (and 934)"""
    ss_regions = mcc.get_structure_tree().get_structures_by_set_id([687527945])

    # 934 not in 100 micron!!!!! (dont want fiber tracts)
    ids, orders = [], []
    for region in ss_regions:
        if region["id"] not in [934, 1009]:
            ids.append(region["id"])
            orders.append(region["graph_order"])

    # return ids sorted by graph order
    ids = np.asarray(ids)
    return ids[np.argsort(orders)]
