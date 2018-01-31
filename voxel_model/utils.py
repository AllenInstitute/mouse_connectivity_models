# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import absolute_import
import numpy as np

from .experiment import Experiment

__all__ = [
    "unique_with_order",
    "lex_ordered_unique_counts",
    "padded_diagonal_fill"
]

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

def padded_diagonal_fill(arrays):
    """stacks uneven arrays padding with zeros"""
    # get a total count of needed columns

    shapes = [x.shape for x in arrays]
    padded = np.zeros( tuple(map(sum, zip(*shapes))) )

    i, j = 0, 0
    for arr in arrays:
        n_rows, n_cols = arr.shape

        padded[i:i+n_rows, j:j+n_cols] = arr

        i += n_rows
        j += n_cols

    return padded
