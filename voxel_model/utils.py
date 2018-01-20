# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import absolute_import
import numpy as np

def get_experiment_ids(mcc, structure_ids, cre=None):
    """Returns all experiment ids given some structure_ids
    PRIMARY INJECTION STRUCTURES
    """
    # filters injections by structure id OR DECENDENT
    experiments = mcc.get_experiments(dataframe=False, cre=cre,
                                      injection_structure_ids=structure_ids)
    return [ experiment['id'] for experiment in experiments ]

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

def column_fill_stack(arrays):
    """stacks uneven arrays padding with zeros"""
    # get a total count of needed columns
    n_cols = np.add.reduce([arr.shape[1] for arr in arrays])

    arrays = [np.hstack( (arr, np.zeros((arr.shape[0], n_cols-arr.shape[1]))) )
              for arr in arrays]

    return np.vstack( arrays )
# def map_descendants(mcc, arr, region_ids_of_interest):
#     """maps arr to regions of interest"""
#     # get list of list of descendants
#     structure_tree = mcc.get_structure_tree()
#     descendant_ids = structure_tree.descendant_ids( region_ids_of_interest )
# 
#     for region_id, descendant_ids in zip(region_ids_of_interest, descendant_ids):
#         # map descendants to roi
#         idx = np.isin(arr, descendant_ids).nonzero()
#         arr[ idx ] = region_id
# 
#     return arr
# 
# def get_lowest_structure_ids(mcc):
#     """Returns only terminal structure_ids"""
#     # get all resolved structures
#     reference_space = mcc.get_reference_space()
#     structures = reference_space.remove_unassigned(update_self=True)
# 
#     # get ids and children ids
#     structure_ids = [ structure["id"] for structure in structures ]
#     child_ids = reference_space.structure_tree.child_ids( structure_ids )
# 
#     # return ids of all
#     return [ sid for sid, cid in zip(structure_ids, child_ids)
#              if len(cid) < 1 ]
# 
# def get_source_master(mcc, structure_ids):
#     # get region ids
#     region_ids = get_lowest_structure_ids(mcc)
# 
#     key = []
#     coordinates = []
#     for sid in structure_ids:
#         build_mask = Mask(mcc, [sid], hemisphere=3)
#         source = Mask(mcc, region_ids, hemisphere=2, other_mask=build_mask)
# 
#         # append to lists
#         key.append(source.key)
#         coordinates.append(source.coordinates)
# 
#     return np.concatenate(key), np.vstack(coordinates)
# 
# def get_target_master(mcc, structure_ids):
#     # get region ids
#     region_ids = get_lowest_structure_ids(mcc)
# 
#     # TODO : IPSI/CONTRA
#     # get mask with which the model was built
#     build_mask = Mask(mcc, structure_ids, hemisphere=3)
#     target = Mask(mcc, region_ids, hemisphere=3, other_mask=build_mask)
# 
#     return target.key, target.coordinates

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
