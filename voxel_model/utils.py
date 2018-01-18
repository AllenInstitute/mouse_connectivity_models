# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import absolute_import
import numpy as np

from .experiment import Experiment

def get_experiment_ids(mcc, structure_ids, cre=None):
    """Returns all experiment ids given some structure_ids
    PRIMARY INJECTION STRUCTURES
    """
    # filters injections by structure id OR DECENDENT
    experiments = mcc.get_experiments(dataframe=False, cre=cre,
                                      injection_structure_ids=structure_ids)
    return [ experiment['id'] for experiment in experiments ]

def get_model_data(mcc, experiment_ids, source_mask, target_mask):
    """..."""
    injections=[]
    projections=[]
    centroids=[]

    for eid in experiment_ids:
        # pull experimental data
        experiment = Experiment(mcc, eid, normalize_projection=True)

        # NEED TO MASK
        centroid = experiment.centroid
        injection = experiment.injection_density[source_mask.where]
        projection = experiment.projection_density[target_mask.where]

        # update
        injections.append(injection)
        projections.append(projection)
        centroids.append(centroid)

    # return arrays
    source_voxels = source_mask.coordinates
    X = np.hstack( (np.asarray(centroids), np.asarray(injections)) )
    y = np.asarray( projections )

    return source_voxels, X, y

def unique_with_order(arr):
    """np.unique with counts in original order."""
    return_params = { "return_index":True, "return_counts":True }
    unique, indices, counts = np.unique(arr, **return_params)

    order = np.argsort(indices)
    return unique[order], counts[order]

def lex_ordered_unique_counts(arr, ordered):
    """np.unique with counts in original order."""
    unique, counts = np.unique(arr, return_counts=True)

    if len(unique) < len(ordered):
        # unique is a subset
        ordered = ordered[ np.isin(ordered, unique) ]

    # return unique and counts ordered by ordered
    order = np.argsort( np.argsort(ordered) )
    return unique[order], counts[order]

def map_descendants(mcc, arr, region_ids_of_interest):
    """maps arr to regions of interest"""
    # get list of list of descendants
    structure_tree = mcc.get_structure_tree()
    descendant_ids = structure_tree.descendant_ids( region_ids_of_interest )

    for region_id, descendant_ids in zip(region_ids_of_interest, descendant_ids):
        # map descendants to roi
        idx = np.isin(arr, descendant_ids).nonzero()
        arr[ idx ] = region_id

    return arr

# def get_permutation_grid(row_ordered, col_ordered,
#                          row_current=None, col_current=None):
#     """Permutes row/columns by ..."""
#     # ensure row_current in row_ordered
#     if row_current is not None:
#         row_ordered = row_ordered[ np.isin(row_ordered, row_current) ]
#     if col_current is not None:
#         col_ordered = col_ordered[ np.isin(col_ordered, col_current) ]
# 
#     # find permutations
#     row = np.argsort( np.argsort(row_ordered) )
#     col = np.argsort( np.argsort(col_ordered) )
# 
#     # return permutation grid
#     return np.ix_(row, col)
