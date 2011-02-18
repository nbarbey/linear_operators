#!/usr/bin/env python

"""
Testing of the lo.ndoperators module
"""

import nose
from numpy.testing import *
import numpy as np
import lo
from lo import ndoperators as op

def generate_permutation(n):
    a = np.arange(n)
    for i in xrange(n - 1, 2, -1):
        j = int(np.floor(np.random.uniform(high=i)))
        a[i], a[j] = a[j], a[i]
    return a

# collection of linear operators to test
shapein16 = (16, 16)
d = np.random.rand(*shapein16)
coef = np.random.uniform(high=10.)
mask = np.random.rand(*shapein16) > .5
s = int(np.floor(np.random.uniform(low=1, high=16)))
p = generate_permutation(16)

id16 = op.ndidentity(shapein16)
d16 = op.nddiagonal(d)
h16 = op.ndhomothetic(shapein16, coef)
m16 = op.ndmask(mask)
f16 = op.fft2(shapein16, dtype=np.complex128)

lo_list = [id16, d16, h16, m16, f16]

# collection of vectors
ones16 = np.ones(shapein16)
zeros16 = np.zeros(shapein16)
rand16 = np.random.rand(*shapein16)

v_list = [ones16, zeros16, rand16 ]

# tests
def check_matvec(A, x):
    A * x.ravel()

def test_matvec():
    for A in lo_list:
        for v in v_list:
            yield check_matvec, A, v

def check_rmatvec(A, x):
    A.T * x.ravel()

def test_rmatvec():
    for A in lo_list:
        for v in v_list:
            yield check_rmatvec, A, v

def check_dense_transpose(A):
    Ad = A.todense()
    Adt = A.T.todense()
    assert_array_equal(np.conj(Ad.T), Adt)

def test_dense_transpose():
    for A in lo_list:
        yield check_dense_transpose, A

if __name__ == "__main__":
    nose.run(argv=['', __file__])
