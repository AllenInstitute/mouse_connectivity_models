"""
Module containing utility functions
"""

# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import absolute_import
from itertools import compress
import numpy as np

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


def get_mcc(manifest_file=None):
    """Returns a MouseConnectivityCache instance with the default settings."""
    if manifest_file is None:
        manifest_file = "connectivity/mouse_connectivity_manifest.json"

    # use 100 micron resolution and the most up to date ccf
    return MouseConnectivityCache(
        manifest_file=manifest_file,
        resolution=100,
        ccf_version="annotation/ccf_2017"
    )


def ordered_unique(arr, return_index=False, return_counts=False, axis=None):
    """np.unique in the order in which the unique values occur.

    Similar outuput to pd.unique(), although probably not as fast.

        see numpy.unique for more info

    Parameters
    ----------
    arr : array
        Array of which unique values are wanted.

    return_index : boolean, optional (default=False)
        If True, first index for each unique value is returned.

    return_counts : boolean, optional (default=False)
        If True, counts of unique values is returned.

    axis : int, optional (defualt=None)
        Axis along which to operate.

    Returns
    -------
    unique : array
        Unique values sorted in the order in which they occur

    unique_indices : array
        Indices of the first occurance of the unique values.

    unique_counts : array
        Counts of the unique values.

    """
    unique = np.unique(arr, return_index=True, return_counts=return_counts, axis=None)

    # unique[1] == indices always
    perm_order = np.argsort(unique[1])
    return_arrs = (True, return_index, return_counts)

    if sum(return_arrs) > 1:
        return tuple(map(lambda x: x[perm_order], compress(unique, return_arrs)))

    return unique[0][perm_order]

def lex_ordered_unique(arr, lex_order, allow_extra=False, return_index=False,
                       return_counts=False, axis=None):
    """np.unique in a given order.

        see numpy.unique for more info

    Parameters
    ----------
    arr : array
        Array of which unique values are wanted.

    lex_order : array, list
        Array describing the order in which the unique values are wanted.

    allow_extra : boolean, optional (default=False)
        If True, lex_order is allowed to have values not found in arr.

    return_index : boolean, optional (default=False)
        If True, first index for each unique value is returned.

    return_counts : boolean, optional (default=False)
        If True, counts of unique values is returned.

    axis : int, optional (defualt=None)
        Axis along which to operate.

    Returns
    -------
    unique : array
        Unique values sorted in the order in which they occur

    unique_indices : array
        Indices of the first occurance of the unique values.

    unique_counts : array
        Counts of the unique values.

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
            lex_order = np.array(lex_order)[np.isin(lex_order, unique[0])]
        else:
            raise ValueError("lex_order contains elements not found in arr, "
                             "call with allow_extra=True")

    # generate a permutation order for unique
    perm_order = np.argsort(np.argsort(lex_order))

    if len(unique) > 1:
        return tuple(map(lambda x: x[perm_order], unique))

    return unique[0][perm_order]

def padded_diagonal_fill(arrays):
    """Returns array filled with uneven arrays padding with zeros.

    Arrays are placed in the return array such that each row/column only
    contains the elements of a single array. Can be thought of as representing
    disconnected subgraphs.

    Parameters
    ----------
    arrays : list
        List of 2D arrays with which to fill the return array.

    Returns
    padded : array
        Return array containing each of the input arrays, padded with zeros.

    """
    shapes = [x.shape for x in arrays]
    padded = np.zeros(tuple(map(sum, zip(*shapes))))

    i, j = 0, 0
    for (n_rows, n_cols), arr in zip(shapes, arrays):
        # fill padded with arr
        padded[i:i+n_rows, j:j+n_cols] = arr

        i += n_rows
        j += n_cols

    return padded
