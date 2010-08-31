#!/usr/bin/env python
""" Test functions for the sparse.linalg.isolve module
"""
import nose

from numpy.testing import *

from numpy import zeros, ones, arange, array, abs, max
from scipy.linalg import norm
from scipy.sparse import spdiags, csr_matrix

from lo.interface import LinearOperator

if __name__ == "__main__":
    nose.run(argv=['', __file__])
