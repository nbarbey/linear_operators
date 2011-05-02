#!/usr/bin/env python

"""
Testing of the lo package
"""

import nose
from numpy.testing import *
import numpy as np
import linear_operators as lo
from linear_operators.iterative import utils

n = 128
k = n - 1
nk = 3
# collection of linear operators to test
I = lo.identity((n, n))
D = lo.diag(1. + np.arange(n))
C = lo.convolve((n,), kernel=1. + np.arange(nk))
lo_list = [I, D, C.T * C]

# collection of vectors
ones16 = np.ones(n)


def check_cond(A):
    Adec = utils.eigendecomposition(A, k=k, which='BE')
    assert_almost_equal(Adec.cond(), np.linalg.cond(A.todense()))

def test_cond():
    for A in lo_list:
        yield check_cond, A

def check_logdet(A):
    Adec = utils.eigendecomposition(A, k=k)
    # very low constraint (decimal = -1)
    assert_almost_equal(Adec.logdet(), np.log(np.linalg.det(A.todense())), decimal=-1)

def test_logdet():
    for A in lo_list:
        yield check_logdet, A

if __name__ == "__main__":
    nose.run(argv=['', __file__])
