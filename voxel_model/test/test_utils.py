import pytest
import operator as op
import numpy as np
from functools import reduce
from numpy.testing import assert_array_equal, assert_raises

from voxel_model.utils \
    import ordered_unique, lex_ordered_unique, padded_diagonal_fill

# -----------------------------------------------------------------------------
# tests:
def test_ordered_unique():
    arr = [0,1,1,2,5,7,1,5,2,9,6,2,6]
    unq = [0,1,2,5,7,9,6]
    idx = list(map(arr.index, unq))
    cnt = list(map(arr.count, unq))

    plain = ordered_unique(arr)
    index = ordered_unique(arr, return_index=True)
    counts = ordered_unique(arr, return_counts=True)
    both = ordered_unique(arr, return_index=True, return_counts=True)

    assert_array_equal( plain, unq )
    for x,y in zip(index, (unq,idx)):
        assert_array_equal(x, y)

    for x,y in zip(counts, (unq,cnt)):
        assert_array_equal(x, y)

    for x,y in zip(both, (unq,idx,cnt)):
        assert_array_equal(x, y)

# -----------------------------------------------------------------------------
# tests:
def test_lex_ordered_unique():
    arr = [0,1,1,2,5,7,1,5,2,9,6,2,6]
    unq = [1,5,7,6,0,9,2]
    idx = list(map(arr.index, unq))
    cnt = list(map(arr.count, unq))

    plain = lex_ordered_unique(arr, unq)
    index = lex_ordered_unique(arr, unq, return_index=True)
    counts = lex_ordered_unique(arr, unq, return_counts=True)
    both = lex_ordered_unique(arr, unq, return_index=True, return_counts=True)

    assert_array_equal( plain, unq )
    for x,y in zip(index, (unq,idx)):
        assert_array_equal(x, y)

    for x,y in zip(counts, (unq,cnt)):
        assert_array_equal(x, y)

    for x,y in zip(both, (unq,idx,cnt)):
        assert_array_equal(x, y)


    # extra value in lex order (not in arr)
    extra = lex_ordered_unique(arr, unq+[10], allow_extra=True)
    assert_array_equal( extra, unq )
    assert_raises( ValueError, lex_ordered_unique, arr, unq+[10] )

    # duplicate
    assert_raises( ValueError, lex_ordered_unique, arr, unq+[0] )

    # assert lex order has not changed
    assert( isinstance(unq, list) )

# -----------------------------------------------------------------------------
# tests:
def test_padded_diagonal_fill():
    a = 1*np.ones((2,5))
    b = 2*np.ones((1,4))
    c = 3*np.ones((3,1))
    d = 4*np.ones((1,1))
    arrs = (a,b,c,d)

    filled = padded_diagonal_fill(arrs)

    assert( filled.shape == (7,11) )
    assert( filled.sum() == reduce(op.add, map(np.sum, arrs)) )

    assert_array_equal( filled[0:2,0:5], a )
    assert_array_equal( filled[2:3,5:9], b )
    assert_array_equal( filled[3:6,9:10], c )
    assert_array_equal( filled[6:7,10:11], d )
