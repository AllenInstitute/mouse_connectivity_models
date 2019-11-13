"""
requires py3
"""
import collections
import itertools
import numpy as np


def inclusive_range(start, stop):
    return tuple(list(range(start, stop)) + [stop])


class NeighborhoodIterator(collections.abc.Iterator):
    """zero padding, only tested in 3D arr"""
    def __init__(self, arr, bounds=None):
        if bounds is not None and len(bounds) != arr.ndim:
            raise ValueError('must pass the same number of bouds as dimensions')

        if bounds is None:
            bounds = [(-1, 1)] * arr.ndim

        self.arr = arr
        self.bounds = bounds

    def __len__(self):
        return self.arr.size

    def __iter__(self):
        self.idx = 0
        return self

    def _get_curr(self):
        return np.unravel_index(self.idx, self.arr.shape)

    def _get_valid_idx(self, idx):
        valid = (idx >= 0) & (idx < np.array(self.arr.shape)[:, np.newaxis])
        return idx[:, valid.all(axis=0)]

    def __next__(self):
        try:
            curr = self._get_curr()
        except IndexError:
            raise StopIteration

        # get relative positions of all in window
        window = [inclusive_range(a, b) + c for (a, b), c in zip(self.bounds, curr)]
        idx = np.vstack(zip(*itertools.product(*window)))

        # do not want index wrapping
        idx = self._get_valid_idx(idx)

        # move iterator
        self.idx += 1

        return idx.tolist()


class MaskedNeighborhoodIterator(NeighborhoodIterator):
    """arr == mask!!!"""
    def __init__(self, arr, bounds=None):
        super().__init__(arr, bounds=bounds)
        self.nnz = np.argwhere(self.arr)

    def _get_curr(self):
        return self.nnz[self.idx]


def masked_filter(arr, mask, key=None, kind='mean', axis=0,
                  bounds=None, filter_kwds=dict()):
    """our application, mask is not binary"""
    if arr.ndim != 2:
        raise ValueError('arr must be 2D')

    if mask.ndim != 3:
        raise ValueError('mask must be 3D')

    if key is None:
        # assume all rows in mask
        key = np.arange(np.count_nonzero(mask))

    # TODO: error handling
    # fill indices in mask (add 1 for nonzero)
    mask = mask.astype(np.int)
    mask[mask.nonzero()] = key + 1

    # filter function
    func = getattr(np, kind)

    result = np.zeros(arr.shape)
    masked_iter = MaskedNeighborhoodIterator(mask, bounds=bounds, **filter_kwds)
    for k, mask_idx in zip(key, masked_iter):

        idx = mask[mask_idx]
        idx = idx[idx.nonzero()] - 1 # we added 1

        filtered = func(arr.take(idx, axis=axis), axis=axis)

        if axis == 0:
            result[k] = filtered
        else:
            result[:, k] = filtered

    return result
