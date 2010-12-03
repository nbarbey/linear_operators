#!/usr/bin/env python

"""
Testing of the lo package
"""

import nose
from numpy.testing import *
import numpy as np
import lo

# collection of linear operators to test
mat16 = lo.aslinearoperator(np.random.rand(16, 16))
lo_list = [mat16, ]

# collection of vectors
ones16 = np.ones(16)

v_list = [ones16, ]

def check_matvec(A, x):
    A * x

def test_matvec():
    for A in lo_list:
        for x in v_list:
            yield check_matvec, A, x

if __name__ == "__main__":
    nose.run(argv=['', __file__])
