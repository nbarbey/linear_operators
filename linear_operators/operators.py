"""Definition of specialized LinearOperator subclasses.

Classes
-------

- SymmetricOperator : Enforces matvec = rmatvec and square shape.

- IdentityOperator : Identity operator.

- HomotheticOperator : Homothetic operators : i.e. of the form a * I.

- DiagonalOperator : Diagonal operator.

- MaskOperator : Diagonal operator with ones and zeros on the diagonal.

- ShiftOperator : Shift the input vector values.

- PermutationOperator : Perform permutations of the input vector values.

- FftOperator : Perform one-dimensional Fast-Fourier Transform (np.fft.fft)

- ReplicationOperator : Replicates the input vector n times.

Functions
---------

Functions generate instances of the associated classes. The following are available :

- identity
- homothetic
- diagonal
- mask
- shift
- permutation
- fft
- replication

"""
import numpy as np
from copy import copy
from interface import LinearOperator, concatenate

# subclasses operators

class SymmetricOperator(LinearOperator):
    """
    Subclass of LinearOperator for the definition of symmetric
    operators, i.e. operators such that : M^T == M.  It only requires
    a shape and a matvec function, since the rmatvec should be the
    same function as matvec.
    """
    def __init__(self, shape, matvec, **kwargs):
        """
        Parameters
        ----------

        shape : length 2 tuple
            The shape of the operator. Should be square.

        matvec : function
            The matrix-vector operation.

        Returns
        -------
        A SymmetricOperator instance.
        """
        if shape[0] != shape[1]:
            raise ValueError("Symmetric operators are square operators.")
        kwargs['rmatvec'] = matvec
        LinearOperator.__init__(self, shape, matvec, **kwargs)

class IdentityOperator(SymmetricOperator):
    def __init__(self, shape, **kwargs):
        """
        Parameters
        ----------

        shape : length 2 tuple
            The shape of the operator. Should be square.

        Returns
        -------
        An IdentityOperator instance.

        Exemple
        -------
        >>> import linear_operators as lo
        >>> I = lo.IdentityOperator((3, 3))
        >>> I.todense()
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]])

        """
        matvec = lambda x: x
        SymmetricOperator.__init__(self, shape, matvec, **kwargs)

class HomotheticOperator(SymmetricOperator):
    def __init__(self, shape, coef, **kwargs):
        """
        Parameters
        ----------

        shape : length 2 tuple
            The shape of the operator. Should be square.

        coef : int or float
            The multiplication coefficient.

        Returns
        -------
        An HomotheticOperator instance.

        Exemple
        -------
        >>> import linear_operators as lo
        >>> H = lo.HomotheticOperator((3, 3), 2.)
        >>> H.todense()
        array([[ 2.,  0.,  0.],
               [ 0.,  2.,  0.],
               [ 0.,  0.,  2.]])

        """
        self.coef = coef
        matvec = lambda x: coef * x
        SymmetricOperator.__init__(self, shape, matvec, **kwargs)
    def __repr__(self):
        s = LinearOperator.__repr__(self)
        return s[:-1] + " and coef=%f >" % self.coef

class DiagonalOperator(SymmetricOperator):
    def __init__(self, d, **kwargs):
        """
        Parameters
        ----------
        d : ndarray with ndim == 1.
            The diagonal of the operator.

        Returns
        -------
        A DiagonalOperator instance.

        Exemple
        -------
        >>> import numpy as np
        >>> import linear_operators as lo
        >>> D = lo.DiagonalOperator(np.arange(3))
        >>> D.todense()
        array([[ 0.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  2.]])
        """
        shape = 2 * (d.size,)
        self.d = d
        matvec = lambda x: x * d
        SymmetricOperator.__init__(self, shape, matvec, **kwargs)
    def __repr__(self):
        s = LinearOperator.__repr__(self)
        return s[:-1] + ",\n diagonal=" + self.d.__repr__() + ">"

class MaskOperator(DiagonalOperator):
    def __init__(self, mask, **kwargs):
        """
        Parameters
        ----------
        mask : ndarray of ones and zeros.
            If mask[i] = 0, the corresponding value will be masked.
            If mask is not a boolean array, it is converted to boolean.

        Returns
        -------
        A MaskOperator instance.

        Exemple
        -------
        >>> import numpy as np
        >>> import linear_operators as lo
        >>> M = lo.MaskOperator(np.arange(4) % 2)
        >>> M.todense()
        array([[ 0.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  1.]])

        """
        self.mask = mask.astype(np.bool)
        DiagonalOperator.__init__(self, self.mask, **kwargs)

class ShiftOperator(LinearOperator):
    def __init__(self, shape, shift, **kwargs):
        if shape[0] != shape[1]:
            raise ValueError("Only square operator is implemented.")
        if np.round(shift) != shift:
            raise ValueError("The shift argument should be an integer value.")
        self.shift = shift
        ashift = np.abs(shift)
        # square case
        matvec = lambda x: np.concatenate((x[ashift:], np.zeros(ashift)))
        rmatvec = lambda x: np.concatenate((np.zeros(ashift), x[:-ashift]))
        if self.shift < 0:
            matvec, rmatvec = rmatvec, matvec
        LinearOperator.__init__(self, shape, matvec, rmatvec=rmatvec, **kwargs)

    def __repr__(self):
        s = LinearOperator.__repr__(self)
        return s[:-1] + " and shift=%d >" % self.shift

class PermutationOperator(LinearOperator):
    def __init__(self, p, **kwargs):
        """
        Parameters
        ----------
        p : list or tuple
            The permutation of coefficients.

        Returns
        -------
        A PermutationOperator instance.

        Exemple
        -------
        >>> import numpy as np
        >>> import linear_operators as lo
        >>> P = lo.PermutationOperator([3, 1, 0, 2])
        >>> P.todense()
        array([[ 0.,  0.,  0.,  1.],
               [ 0.,  1.,  0.,  0.],
               [ 1.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  0.]])

        """

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
    def __init__(self, shape, n, **kwargs):
        """
        Parameters
        ----------
        shape : length 2 tuple.
            The shape of the operator.

        n : int
            The number of replications.

        Returns
        -------
        A ReplicationOperator instance.

        Exemple
        -------
        >>> import numpy as np
        >>> import linear_operators as lo
        >>> R = lo.ReplicationOperator((4, 2), 2)
        >>> R.todense()
        array([[ 1.,  0.],
               [ 0.,  1.],
               [ 1.,  0.],
               [ 0.,  1.]])
        """
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

class SliceOperator(LinearOperator):
    def __init__(self, shape, slice, **kwargs):
        def matvec(x):
            return x[slice]
        def rmatvec(x):
            out = np.zeros(shape[1])
            out[slice] = x
            return out
        LinearOperator.__init__(self, shape, matvec, rmatvec=rmatvec, **kwargs)

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

def slice(shape, slice, **kwargs):
    return SliceOperator(shape, slice, **kwargs)
