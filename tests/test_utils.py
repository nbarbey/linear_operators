#!/usr/bin/env python

"""
Testing of the lo package
"""

import nose
from numpy.testing import *
import numpy as np
import lo
from lo.iterative.utils import cond

# collection of linear operators to test
D = lo.diag(1. + np.arange(16))
I = lo.identity((16, 16))
C = lo.convolve((8,), 1. + np.arange(8))
lo_list = [D, I, C.T * C]

# collection of vectors
ones16 = np.ones(16)


def check_cond(A):
    assert_almost_equal(cond(lo.aslinearoperator(A)), np.linalg.cond(A.todense()))

def test_cond():
    for A in lo_list:
        yield check_cond, A

if __name__ == "__main__":
    nose.run(argv=['', __file__])
