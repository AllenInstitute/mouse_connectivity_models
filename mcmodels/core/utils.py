"""
Module containing utility functions for the :mod:`mcmodels.core` module.
"""
# Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License

from __future__ import division

import numpy as np



def get_matrices(experiments):
    get_data = lambda x: (get_centroid(x),
                          get_injection(x,True),
                          get_projection(x, True))
    arrays = map(get_data, yield_experiments(experiments))
    centroids, injections, projections = map(np.array, zip(*arrays))
    return(centroids, injections, projections)

def get_centroid(experiment):
    """Returns experiment centroid"""
    return experiment.centroid

def get_injection(experiment, normalized_injection):
    # print('ts',experiment.normalized_injection)
    """Returns experiment injection masked & flattened"""
    injection = experiment.get_injection(normalized_injection)
    return experiment.injection_mask.mask_volume(injection)

def get_projection(experiment, normalized_projection):
    """Returns experiment projection masked & flattened"""
    projection = experiment.get_projection(normalized_projection)
    return experiment.projection_mask.mask_volume(projection)

def yield_experiments(experiments):
    ev = experiments.values()
    keys = np.asarray(list(experiments.keys()))
    for i in range(len(keys)):
        yield (experiments[keys[i]])

def get_loss_paper(y,yhat):
    loss = 2* np.linalg.norm(y - yhat)**2 / (np.linalg.norm(y)**2 + np.linalg.norm(yhat)**2)
    return(loss)


def get_structure_id(cache, acronym):
    try:
        return cache.get_structure_tree().get_structures_by_acronym([acronym])[0]['id']
    except KeyError:
        raise ValueError("structure acronym (%s) is not valid" % acronym)




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


def get_minorstructures(eids, data_info):
    #eids = np.asarray(list(msvd.experiments.keys()))
    experiments_minors = np.zeros(len(eids), dtype=object)

    for i in range(len(eids)):
        experiment_id = eids[i]
        experiments_minors[i] = data_info['primary-injection-structure'].loc[experiment_id]

    return (experiments_minors)


####old

def compute_centroid(injection_density):
    """Computes centroid in index coordinates.

    Parameters
    ----------
    injection_density : array, shape (x_ccf, y_ccf, z_ccf)
        injection_density data volume.

    Returns
    -------
        centroid of injection_density in index coordinates.
    """
    nnz = injection_density.nonzero()
    coords = np.vstack(nnz)

    return np.dot(coords, injection_density[nnz]) / injection_density.sum()


def get_injection_hemisphere_id(injection_density, majority=False):
    """Gets injection hemisphere based on injection density.

    Defines injection hemisphere by the ratio of the total injection_density
    in each hemisphere.

    Parameters
    ----------
    injection_density : array, shape (x_ccf, y_ccf, z_ccf)
        injection_density data volume.

    Returns
    -------
    int : in (1,2,3)
        injection_hemisphere
    """
    if injection_density.ndim != 3:
        raise ValueError("injection_density must be 3-array not (%d)-array"
                         % injection_density.ndim)

    # split along depth dimension (forces arr.shape[2] % 2 == 0)
    hemis = np.dsplit(injection_density, 2)
    hemi_sums = tuple(map(np.sum, hemis))

    # if not looking for either l or r
    if not majority and all(hemi_sums):
        return 3

    left_sum, right_sum = hemi_sums
    if left_sum > right_sum:
        return 1

    return 2
