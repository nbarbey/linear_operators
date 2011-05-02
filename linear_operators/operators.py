"""Definition of useful linear operators"""
import numpy as np
from copy import copy
from interface import LinearOperator, concatenate

# subclasses operators

class SymmetricOperator(LinearOperator):
    def __init__(self, shape, matvec, **kwargs):
        if shape[0] != shape[1]:
            raise ValueError("Symmetric operators are square operators.")
        kwargs['rmatvec'] = matvec
        LinearOperator.__init__(self, shape, matvec, **kwargs)

class IdentityOperator(SymmetricOperator):
    def __init__(self, shape, **kwargs):
        matvec = lambda x: x
        SymmetricOperator.__init__(self, shape, matvec, **kwargs)

class HomotheticOperator(SymmetricOperator):
    def __init__(self, shape, coef, **kwargs):
        self.coef = coef
        matvec = lambda x: coef * x
        SymmetricOperator.__init__(self, shape, matvec, **kwargs)
    def __repr__(self):
        s = LinearOperator.__repr__(self)
        return s[:-1] + " and coef=%f >" % self.coef

class DiagonalOperator(SymmetricOperator):
    def __init__(self, d, **kwargs):
        shape = 2 * (d.size,)
        self.d = d
        matvec = lambda x: x * d
        SymmetricOperator.__init__(self, shape, matvec, **kwargs)
    def __repr__(self):
        s = LinearOperator.__repr__(self)
        return s[:-1] + ",\n diagonal=" + self.d.__repr__() + ">"

class MaskOperator(DiagonalOperator):
    def __init__(self, mask, **kwargs):
        self.mask = mask.astype(np.bool)
        DiagonalOperator.__init__(self, self.mask, **kwargs)

class ShiftOperator(LinearOperator):
    def __init__(self, shape, shift, **kwargs):
        self.shift = shift
        matvec = lambda x: np.concatenate((x[shift:], np.zeros(shift)))
        rmatvec = lambda x: np.concatenate((np.zeros(shift), x[:-shift]))
        if shape[0] <= shape[1]:
            def matvec(x):
                return np.concatenate((x[shift:], np.zeros(shift)))[:shape[0]]
            def rmatvec(x):
                tmp = np.zeros(shape[1])
                tmp[:shape[0]] = np.concatenate((np.zeros(shift), x[:-shift]))
                return tmp
        if shape[0] >= shape[1]:
            def matvec(x):
                tmp = np.zeros(shape[0])
                tmp[:shape[1]] = np.concatenate((np.zeros(shift), x[:-shift]))
                return tmp
            def rmatvec(x):
                return np.concatenate((x[shift:], np.zeros(shift)))[:shape[1]]
        LinearOperator.__init__(self, shape, matvec, rmatvec=rmatvec, **kwargs)
    def __repr__(self):
        s = LinearOperator.__repr__(self)
        return s[:-1] + " and shift=%d >" % self.shift

class PermutationOperator(LinearOperator):
    def __init__(self, p, **kwargs):
        shape = 2 * (len(p),)
        self.p = p
        # reverse permutation
        self.p_inv = np.argsort(p)
        def matvec(x):
            return np.asarray([x[pi] for pi in self.p])
        def rmatvec(x):
            return np.asarray([x[pi] for pi in self.p_inv])
        LinearOperator.__init__(self, shape, matvec, rmatvec=rmatvec, **kwargs)
    def __repr__(self):
        s = LinearOperator.__repr__(self)
        return s[:-1] + ",\n permutation=" + self.p.__repr__() + ">"

class FftOperator(LinearOperator):
    def __init__(self, shape, **kwargs):
        matvec = lambda x: np.fft.fft(x, n=shape[0]) / np.sqrt(shape[0])
        rmatvec = lambda x: np.fft.ifft(x, n=shape[1]) * np.sqrt(shape[0])
        LinearOperator.__init__(self, shape, matvec, rmatvec=rmatvec, **kwargs)

class ReplicationOperator(LinearOperator):
    """
    Output n times the input
    """
    def __init__(self, shape, n, **kwargs):
        self.n = n
        # ensure coherent input
        if not shape[0] == shape[1] * n:
            raise ValueError("Output vector should be n times the size of input vector.")
        def matvec(x):
            return np.tile(x, n)
        def rmatvec(x):
            N = shape[1]
            return sum([x[i * N:(i + 1) * N] for i in xrange(n)])
        LinearOperator.__init__(self, shape, matvec, rmatvec=rmatvec, **kwargs)
    def __repr__(self):
        s = LinearOperator.__repr__(self)
        return s[:-1] + ",\n Replicate %i times" % self.n + ">"

# functions
def identity(shape, **kwargs):
    """
    Parameters:
    shape : the shape of the operator
    Output:
    I : identity LinearOperator

    Exemple:
    >>> I = identity((16, 16), dtype=float32)
    """
    return IdentityOperator(shape, **kwargs)

def homothetic(shape, coef, **kwargs):
    return HomotheticOperator(shape, coef, **kwargs)

def diagonal(d, **kwargs):
    return DiagonalOperator(d, **kwargs)

# for backward compatibility
diag = diagonal

def mask(mask, **kwargs):
    return MaskOperator(mask, **kwargs)

def shift(shape, s, **kwargs):
    return ShiftOperator(shape, s, **kwargs)

def permutation(p, **kwargs):
    return PermutationOperator(p, **kwargs)

def fft(shape, **kwargs):
    return FftOperator(shape, **kwargs)

def replication(shape, n, **kwargs):
    return ReplicationOperator(shape, n, **kwargs)
