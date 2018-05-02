"""
Module containing utility functions
"""
# Authors: Joseph Knox josephk@alleninstitute.org
# License: Allen Institute Software License

import numpy as np
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


def get_experiment_ids(mcc, structure_ids, cre=None):
    """Returns all experiment ids with injection in structure_ids.

    Parameters
    ----------
    mcc : MouseConnectivityCache object
    structure_ids: list
        Only return experiments that were injected in the structures provided here.
        If None, return all experiments.  Default None.
    cre: boolean or list
        If True, return only cre-positive experiments.  If False, return only
        cre-negative experiments.  If None, return all experiments. If list, return
        all experiments with cre line names in the supplied list. Default None.

    Returns
    -------
    List of experiment ids satisfying the parameters.
    """
    #filters injections by structure id or Decendent
    experiments = mcc.get_experiments(dataframe=False, cre=cre,
                                      injection_structure_ids=structure_ids)
    return [experiment['id'] for experiment in experiments]


def nonzero_unique(ar, **unique_kwargs):
    """np.unique returning only nonzero unique elements.

    Parameters
    ----------
    ar : array_like
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.

    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.

    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.

    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened beforehand.
        Otherwise, duplicate items will be removed along the provided axis,
        with all the other axes belonging to the each of the unique elements.
        Object arrays or structured arrays that contain objects are not
        supported if the `axis` kwarg is used.

    Returns
    -------
    unique : array
        Unique values sorted in the order in which they occur

    unique_indices : array, optional
        Indices of the first occurance of the unique values. Only returned if
        return_indices kwarg is specified as True.

    unique_counts : array
        Counts of the unique values. Only returned if return_counts kwarg is
        specified as True.

    See Also
    --------
    ordered_unique
    lex_ordered_unique
    """
    if 'return_inverse' in unique_kwargs:
        raise NotImplementedError("returning inverse array not yet implemented")

    if np.all(ar):
        return np.unique(ar, **unique_kwargs)

    unique = np.unique(ar, **unique_kwargs)
    if unique_kwargs:
        return map(lambda x: x[1:], unique)

    return unique[1:]


def ordered_unique(ar, **unique_kwargs):
    """np.unique in the order in which the unique values occur.

    Similar outuput to pd.unique(), although probably not as fast.

    Parameters
    ----------
    ar : array_like
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.

    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.

    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.

    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened beforehand.
        Otherwise, duplicate items will be removed along the provided axis,
        with all the other axes belonging to the each of the unique elements.
        Object arrays or structured arrays that contain objects are not
        supported if the `axis` kwarg is used.

    Returns
    -------
    unique : array
        Unique values sorted in the order in which they occur

    unique_indices : array, optional
        Indices of the first occurrence of the unique values. Only returned if
        return_indices kwarg is specified as True.

    unique_counts : array
        Counts of the unique values. Only returned if return_counts kwarg is
        specified as True.

    See Also
    --------
    nonzero_unique
    lex_ordered_unique
    """
    if 'return_inverse' in unique_kwargs:
        raise NotImplementedError("returning inverse array not yet implemented")

    _return_index = unique_kwargs.pop('return_index', False)
    unique = np.unique(ar, return_index=True, **unique_kwargs)

    # need indices (always @ index 1)
    unique = list(unique)
    indices = unique[1] if _return_index else unique.pop(1)
    permutation = np.argsort(indices)

    if unique_kwargs or _return_index:
        return map(lambda x: x[permutation], unique)

    return unique[0][permutation]


def lex_ordered_unique(ar, lex_order, allow_extra=False, **unique_kwargs):
    """np.unique in a given order.

    Parameters
    ----------
    ar : array_like
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.

    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.

    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.

    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened beforehand.
        Otherwise, duplicate items will be removed along the provided axis,
        with all the other axes belonging to the each of the unique elements.
        Object arrays or structured arrays that contain objects are not
        supported if the `axis` kwarg is used.

    Returns
    -------
    unique : array
        Unique values sorted in the order in which they occur

    unique_indices : array, optional
        Indices of the first occurrence of the unique values. Only returned if
        return_indices kwarg is specified as True.

    unique_counts : array
        Counts of the unique values. Only returned if return_counts kwarg is
        specified as True.

    See Also
    --------
    nonzero_unique
    ordered_unique
    """
    if 'return_inverse' in unique_kwargs:
        raise NotImplementedError("returning inverse array not yet implemented")

    if len(set(lex_order)) < len(lex_order):
        raise ValueError("lex_order must not contain duplicates")

    unique = np.unique(ar, **unique_kwargs)
    if not unique_kwargs:
        unique = (unique,)

    if len(unique[0]) < len(lex_order):
        if allow_extra:
            # view, does not write to array lex_order
            # cast to np.array in order to index with boolean array
            lex_order = np.array(lex_order)[np.isin(lex_order, unique[0])]
        else:
            raise ValueError("lex_order contains elements not found in ar, "
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
    -------
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
    """Unionize voxel data to regional data.

    Parameters
    ----------
    volume : array, shape (n, m)
        Possibly stacked flattened volume(s) such as projection densities or
        injection densities.
    key : array, shape (m,)
        1D Array with length equal to the number of columns in volume. This array
        has integer values corresponding to the region to which each voxel belongs.
    return_regions : boolean, optional (default: False)
        If True, return an array of the unique values of key in addition to the
        unionized volume array.

    Returns
    -------
    result : array, shape (n, len(unique(key)))
        The unionized volume.
    regions : array, optional, shape (len(unique(key)),)
        The unique values of key.
    """
    volume = np.atleast_2d(volume)
    if volume.shape[1] != key.size:
        raise ValueError("volume (%s) and key (%s) shapes are incompatible"
                         % (volume.shape[1], key.size))

    regions = nonzero_unique(key)
    result = np.empty((volume.shape[0], regions.size))
    for j, k in enumerate(regions):
        result[:, j] = volume[:, np.where(key == k)[0]].sum(axis=1)

    if return_regions:
        return result, regions

    return result
