"""Test functions for the lo.operators module
"""
import nose
import copy

from numpy.testing import *

import numpy as np
import scipy.sparse as sparse

from lo.operators import *

class TestNDOperator(TestCase):
    def setUp(self):
        self.vars = []
        shapein1 = (3, 4)
        shapeout1 = (3,)
        def ndmatvec1(x):
            return np.array([ 1*x[0, 0] + 2*x[1, 0] + 3*x[2, 0],
                              4*x[0, 1] + 5*x[0, 2] + 6*x[0, 3],
                              7*x[1, 1] + 8*x[1, 2] + 9*x[2, 3]])
        def ndrmatvec1(x):
            y = np.zeros((3, 4))
            y[0, 0] = 1 * x[0]
            y[1, 0] = 2 * x[0]
            y[2, 0] = 3 * x[0]
            y[0, 1] = 4 * x[1]
            y[0, 2] = 5 * x[1]
            y[0, 3] = 6 * x[1]
            y[1, 1] = 7 * x[2]
            y[1, 2] = 8 * x[2]
            y[2, 3] = 9 * x[2]
            return y

        Ad1 = np.zeros((np.prod(shapeout1), np.prod(shapein1)))
        Ad1[0, 0] = 1
        Ad1[0, 4] = 2
        Ad1[0, 8] = 3
        Ad1[1, 1] = 4
        Ad1[1, 2] = 5
        Ad1[1, 3] = 6
        Ad1[2, 5] = 7
        Ad1[2, 6] = 8
        Ad1[2, 11] = 9

        self.vars.append([shapein1, shapeout1, ndmatvec1, ndrmatvec1, Ad1])

    def test_ndoperator(self):
        for var in self.vars:
            # initialization
            shapein, shapeout, ndmatvec, ndrmatvec, Ad = var
            A = ndoperator(shapein, shapeout, ndmatvec, ndrmatvec)
            # tests
            assert_equal(A.shape, (np.prod(shapeout), np.prod(shapein)))
            assert_equal(A * np.ones(np.prod(shapein)), np.array([6, 15, 24]))
            assert_equal(A * np.zeros(np.prod(shapein)), np.array([0, 0, 0]))
            # dense
            assert_equal(A.todense(), Ad)
            assert_equal(A.T.todense(), Ad.T)

class TestMASubclass(TestCase):
    def setUp(self):
        self.vars = []
        shapein1 = (3, 4)
        shapeout1 = (3,)
        def ndmatvec1(x):
            if not isinstance(x, np.ma.MaskedArray):
                raise ValueError('Expected MaskedArray')
            return np.ma.asarray([ 1*x[0, 0] + 2*x[1, 0] + 3*x[2, 0],
                                   4*x[0, 1] + 5*x[0, 2] + 6*x[0, 3],
                                   7*x[1, 1] + 8*x[1, 2] + 9*x[2, 3]])
        def ndrmatvec1(x):
            if not isinstance(x, np.ma.MaskedArray):
                raise ValueError('Expected MaskedArray')
            y = np.ma.zeros((3, 4))
            y[0, 0] = 1 * x[0]
            y[1, 0] = 2 * x[0]
            y[2, 0] = 3 * x[0]
            y[0, 1] = 4 * x[1]
            y[0, 2] = 5 * x[1]
            y[0, 3] = 6 * x[1]
            y[1, 1] = 7 * x[2]
            y[1, 2] = 8 * x[2]
            y[2, 3] = 9 * x[2]
            return y

        Ad1 = np.zeros((np.prod(shapeout1), np.prod(shapein1)))
        Ad1[0, 0] = 1
        Ad1[0, 4] = 2
        Ad1[0, 8] = 3
        Ad1[1, 1] = 4
        Ad1[1, 2] = 5
        Ad1[1, 3] = 6
        Ad1[2, 5] = 7
        Ad1[2, 6] = 8
        Ad1[2, 11] = 9

        self.vars.append([shapein1, shapeout1, ndmatvec1, ndrmatvec1, Ad1])
    def test_masubclass(self):
        for var in self.vars:
            # initialization
            shapein, shapeout, ndmatvec, ndrmatvec, Ad = var
            xin = np.ma.MaskedArray(np.zeros(shapein))
            xout = np.ma.MaskedArray(np.zeros(shapeout))
            A = masubclass(xin=xin, xout=xout, matvec=ndmatvec, rmatvec=ndrmatvec)
            # tests
            assert_equal(A.shape, (np.prod(shapeout), np.prod(shapein)))
            assert_equal(A * np.ma.ones(np.prod(shapein)), np.ma.asarray([6, 15, 24]))
            assert_equal(A * np.ma.zeros(np.prod(shapein)), np.ma.asarray([0, 0, 0]))
            # dense
            assert_equal(A.todense(), Ad)
            assert_equal(A.T.todense(), Ad.T)

class InfoArray(np.ndarray):
    """An ndarray supplemented with an header object.

    There is no requirement on the nature of the object
    """
    def __new__(subtype, shape=None, data=None, dtype=float, buffer=None, offset=0,
                strides=None, order=None, header=None):
        if shape is not None:
            obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset,
                                     strides, order)
        elif data is not None:
            obj = np.array(data).view(subtype)
        obj.header = header
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.header = getattr(obj, 'header', None)
    def copy(self):
        return asinfoarray(copy.copy(self), header=copy.copy(self.header))

def asinfoarray(array, header=None):
    """Return a view of an array as a FitsArray
    """
    return InfoArray(data=array, header=header)

class TestNDSubclass(TestCase):
    def setUp(self):
        self.vars = []
        shapein1 = (3, 4)
        shapeout1 = (3,)

        def ndmatvec1(x):
            if not isinstance(x, InfoArray):
                raise ValueError('Expected InfoArray')
            y = asinfoarray([1 * x[0, 0] + 2 * x[1, 0] + 3 * x[2, 0],
                             4 * x[0, 1] + 5 * x[0, 2] + 6 * x[0, 3],
                             7 * x[1, 1] + 8 * x[1, 2] + 9 * x[2, 3]])
            print x.header
            if x.header is not None:
                y *= x.header['prod']
            return y
                
        def ndrmatvec1(x):
            if not isinstance(x, InfoArray):
                raise ValueError('Expected InfoArray')
            y = asinfoarray(np.zeros((3, 4)))
            y[0, 0] = 1 * x[0]
            y[1, 0] = 2 * x[0]
            y[2, 0] = 3 * x[0]
            y[0, 1] = 4 * x[1]
            y[0, 2] = 5 * x[1]
            y[0, 3] = 6 * x[1]
            y[1, 1] = 7 * x[2]
            y[1, 2] = 8 * x[2]
            y[2, 3] = 9 * x[2]
            if x.header is not None:
                y *= x.header['prod']
            return y

        Ad1 = np.zeros((np.prod(shapeout1), np.prod(shapein1)))
        Ad1[0, 0] = 1
        Ad1[0, 4] = 2
        Ad1[0, 8] = 3
        Ad1[1, 1] = 4
        Ad1[1, 2] = 5
        Ad1[1, 3] = 6
        Ad1[2, 5] = 7
        Ad1[2, 6] = 8
        Ad1[2, 11] = 9

        self.vars.append([shapein1, shapeout1, ndmatvec1, ndrmatvec1, Ad1])

    def test_ndsubclass(self):
        for var in self.vars:
            # initialization
            shapein, shapeout, ndmatvec, ndrmatvec, Ad = var
            header = None
            xin = asinfoarray(np.ones(shapein), header)
            xout = asinfoarray(np.ones(shapeout), header)
            A = ndsubclass(xin=xin, xout=xout, matvec=ndmatvec, rmatvec=ndrmatvec)
            # tests
            assert_equal(A.shape, (np.prod(shapeout), np.prod(shapein)))
            assert_equal(A * np.ones(np.prod(shapein)), asinfoarray([6, 15, 24]))
            assert_equal(A * np.zeros(np.prod(shapein)), asinfoarray([0, 0, 0]))
            # dense
            assert_equal(A.todense(), Ad)
            assert_equal(A.T.todense(), Ad.T)
            # header
            header = {'prod':2}
            xin = asinfoarray(np.ones(shapein), header)
            xout = asinfoarray(np.ones(shapeout), header)
            A = ndsubclass(xin=xin, xout=xout, matvec=ndmatvec, rmatvec=ndrmatvec)
            assert_equal(A * np.ones(np.prod(shapein)), asinfoarray([12, 30, 48]))
            assert_equal(A.T * np.zeros(np.prod(shapeout)), np.zeros(np.prod(shapein)))

if __name__ == "__main__":
    nose.run(argv=['', __file__])
