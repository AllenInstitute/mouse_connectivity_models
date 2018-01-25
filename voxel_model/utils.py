# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import absolute_import
import numpy as np
from collections import namedtuple

from .experiment import Experiment

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
