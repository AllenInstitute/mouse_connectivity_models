"""
Module containing utility functions
"""

# Authors: Joseph Knox josephk@alleninstitute.org
# License:

from __future__ import absolute_import
from itertools import compress
import os

import numpy as np
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


def get_experiment_ids(mcc, structure_ids, cre=None):
    """Returns all experiment ids with injection in ..."""
    #TODO: improve doc
    #filters injections by structure id or Decendent
    experiments = mcc.get_experiments(dataframe=False, cre=cre,
                                      injection_structure_ids=structure_ids)
    return [experiment['id'] for experiment in experiments]

def get_mcc(manifest_file=None):
    """Returns a MouseConnectivityCache instance with the default settings."""
    if manifest_file is None:
        manifest_file = os.path.join(os.getcwd(), "connectivity",
                                     "mouse_connectivity_manifest.json")

    # use 100 micron resolution and the most up to date ccf
    return MouseConnectivityCache(
        manifest_file=manifest_file,
        resolution=100,
        ccf_version=os.path.join("annotation", "ccf_2017")
    )


def nonzero_unique(array, **unique_kwargs):
    # TODO: docstring
    if 'return_inverse' in unique_kwargs:
        raise NotImplementedError("lex ordiring of inverse array not "
                                  "yet implemented")

    if np.all(array):
        return np.unique(array, **unique_kwargs)

    unique = np.unique(array, **unique_kwargs)
    if unique_kwargs:
        return map(lambda x: x[1:], unique)

    return unique[1:]


def ordered_unique(array, **unique_kwargs):
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
    if 'return_inverse' in unique_kwargs:
        raise NotImplementedError("lex ordiring of inverse array not "
                                  "yet implemented")

    _return_index = unique_kwargs.pop('return_index', False)
    unique = np.unique(array, return_index=True, **unique_kwargs)

    # need indices (always @ index 1)
    unique = list(unique)
    indices = unique[1] if _return_index else unique.pop(1)
    permutation = np.argsort(indices)

    if unique_kwargs or _return_index:
        return map(lambda x: x[permutation], unique)

    return unique[0][permutation]


def lex_ordered_unique(arr, lex_order, allow_extra=False, **unique_kwargs):
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
    if 'return_inverse' in unique_kwargs:
        raise NotImplementedError("lex ordiring of inverse array not "
                                  "yet implemented")

    if len(set(lex_order)) < len(lex_order):
        raise ValueError("lex_order must not contain duplicates")

    unique = np.unique(arr, **unique_kwargs)
    if not unique_kwargs:
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
    permutation = np.argsort(np.argsort(lex_order))

    if len(unique) > 1:
        return tuple(map(lambda x: x[permutation], unique))

    return unique[0][permutation]


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


def unionize(volume, key, return_regions=False):
    """Unionize voxel data to regional data"""
    volume = np.atleast_2d(volume)
    if volume.shape[1] != key.size:
        # TODO: better error
        raise ValueError("key is incompatible")

    regions = nonzero_unique(key)
    result = np.empty((volume.shape[0], regions.size))
    for j, k in enumerate(regions):
        result[:, j] = volume[:, np.where(key == k)[0]].sum(axis=1)

    if return_regions:
        return result, regions
    return result
