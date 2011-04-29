#!/usr/bin/env python

"""
Testing of the lo.iterative module
"""

import nose
from numpy.testing import *
import numpy as np
import lo
from lo import iterative

# collection of linear operators to test
mat16 = lo.aslinearoperator(np.random.rand(16, 16))
id16 = lo.identity((16, 16))
diag16 = lo.diag(np.random.rand(16))
conv16 = lo.convolve(16, np.random.rand(4), mode="same")
lo_list = [mat16, id16, diag16, conv16]

# collection of vectors
ones16 = np.ones(16)
zeros16 = np.zeros(16)
rand16 = np.random.rand(16)

v_list = [ones16, zeros16, rand16 ]

# collection of methods
methods = [iterative.acg, ]

# tests
def check_inv(method, A, x):
    y = A * x
    xe = method(A, y, maxiter=100, tol=1e-6)
    assert_almost_equal, x, xe

def test_inv():
    for A in lo_list:
        for x in v_list:
            for m in methods:
                yield check_inv, m, A, x

#def check_rls_vs_acg(A, x):
#    y = A * x
#    x_rls = iterative.rls(A, y, maxiter=100, tol=1e-6)
#    x_acg = iterative.acg(A, y, maxiter=100, tol=1e-6)
#    assert_almost_equal, x_rls, x_acg

#def test_rls_vs_acg():
#    for A in lo_list:
#        for x in v_list:
#            yield check_rls_vs_acg, A, x

if __name__ == "__main__":
    nose.run(argv=['', __file__])
