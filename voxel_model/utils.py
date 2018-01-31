# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import absolute_import
from itertools import compress
import numpy as np

from .experiment import Experiment

__all__ = [
    "ordered_unique",
    "lex_ordered_unique",
    "padded_diagonal_fill"
]

def ordered_unique(arr, return_index=False, return_counts=False, axis=None):
    """np.unique with counts in the order inwhich they occur.

    Similar outuput to pd.unique(), althouhg probably not as fast.

    """
    unique = np.unique( arr, return_index=True, return_counts=return_counts,
                        axis=None )

    # unique[1] == indices always
    perm_order = np.argsort(unique[1])
    return_arrs = (True, return_index, return_counts)

    if sum(return_arrs) > 1:
        return tuple( map(lambda x: x[perm_order],
                          compress(unique, return_arrs)) )
    else:
        return unique[0][perm_order]

def lex_ordered_unique(arr, lex_order, allow_extra=False, return_index=False,
                       return_counts=False, axis=None):
    """np.unique with counts in given lexiconigraphic order.

    ...

    Parameters
    """
    if len(set(lex_order)) < len(lex_order):
        raise ValueError("lex_order must not contain duplicates")

    unique = np.unique(arr, return_index=return_index,
                       return_counts=return_counts, axis=axis)

    if not return_index and not return_counts:
        unique = (unique,)

    if len(unique[0]) < len(lex_order):
        if allow_extra:
            # view, does not write to array lex_order
            # cast to np.array in order to index with boolean array
            lex_order = np.array(lex_order)[ np.isin(lex_order, unique[0]) ]
        else:
            raise ValueError( "lex_order contains elements not found in arr ",
                              "call with allow_extra=True" )

    # generate a permutation order for unique
    perm_order = np.argsort( np.argsort(lex_order) )

    if len(unique) > 1:
        return tuple(map(lambda x: x[perm_order], unique))
    else:
        return unique[0][perm_order]

def padded_diagonal_fill(arrays):
    """stacks uneven arrays padding with zeros"""

    shapes = [x.shape for x in arrays]
    padded = np.zeros( tuple(map(sum, zip(*shapes))) )

    i, j = 0, 0
    for (n_rows, n_cols), arr in zip(shapes, arrays):
        # fill padded with arr
        padded[i:i+n_rows, j:j+n_cols] = arr

        i += n_rows
        j += n_cols

    return padded
