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
        self.idx = 0

    def __len__(self):
        return self.arr.size

    def __iter__(self):
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

        # TODO: remove tolist()
        return idx.tolist()


class MaskedNeighborhoodIterator(NeighborhoodIterator):
    """arr == mask!!!"""
    def __init__(self, arr, bounds=None):
        super().__init__(arr, bounds=bounds)
        self.nnz = np.argwhere(self.arr)

    def _get_curr(self):
        return self.nnz[self.idx]
