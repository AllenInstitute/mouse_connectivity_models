import pytest
import numpy as np
from functools import reduce
from numpy.testing import assert_array_equal

from voxel_model.utils \
    import unique_with_order, lex_ordered_unique_counts, column_fill_stack

# -----------------------------------------------------------------------------
# tests:
def test_unique_with_order():
    pass

# -----------------------------------------------------------------------------
# tests:
def test_lex_ordered_unique_counts():
    pass

# -----------------------------------------------------------------------------
# tests:
def test_column_fill_stack():
    a = 1*np.ones((2,5))
    b = 2*np.ones((1,4))
    c = 3*np.ones((3,1))
    d = 4*np.ones((1,1))

    filled = column_fill_stack([a,b,c,d])

    assert( filled.shape == (7,11) )
    assert( filled.sum() == np.add.reduce(map(np.sum, [a,b,c,d])) )

    assert_array_equal( filled[0:2,0:5], a )
    assert_array_equal( filled[2:3,5:9], b )
    assert_array_equal( filled[3:6,9:10], c )
    assert_array_equal( filled[6:7,10:11], d )
