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

- SliceOperator : Performs slicing on an array.

- TridiagonalOperator : A tridiagonal matrix a 3 1d arrays.

- SymmetricTridiagonal : A symmetric tridiagonal matrix a 2 1d arrays.

- BandOperator : A band matrix using lapack storage scheme.

- SymmetricBandOperator : A symmetric band matrix using lapack storage scheme.

- LoweTriangularOperator : A special case of band operator with only subdiagonal bands.

- UpperTriangularOperator : A special case of band operator with only superdiagonal bands.

- EigendecompositionOperator : A class storing eigenpairs of another operator.

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
- slice_operator
- tridiagonal

- band_approximation : Generate a BandOperator from a dense matrix.
- symmetric_band_approximation : Generate a BandOperator from a dense matrix.

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
        """Returns a SymmetricOperator of given shape and matvec
        function.

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
    """SymmetricOperator with identity matvec (lambda x: x).
    """
    def __init__(self, shape, **kwargs):
        """Instantiate an IdentityOperator.

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
    """
    Generate a SymmetricOperator performing an homothety, i.e. a
    multiplication of the input vector by a scalar.

    Arguments
    ----------
    coef : int float
        Multiplication coefficient
    """
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
    """
    An operator which mimics a diagonal matrix.

    Attributes
    ----------
    diag: ndarray of 1 dim
       The diagonal of the matrix.
    """
    def __init__(self, diag, **kwargs):
        """
        Parameters
        ----------
        diag : ndarray with ndim == 1.
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
        shape = 2 * (diag.size,)
        self.diag = diag
        matvec = lambda x: x * diag
        SymmetricOperator.__init__(self, shape, matvec, **kwargs)
    def __repr__(self):
        s = LinearOperator.__repr__(self)
        return s[:-1] + ",\n diagonal=" + self.diag.__repr__() + ">"

class MaskOperator(DiagonalOperator):
    """
    A subclass of DiagonalOperator with a boolean diagonal. Elements
    corresponding to zeros of the diagonal are masked out in the
    output vector.

    Attributes
    -----------
    mask : ndarray of type bool and ndim==1
    """
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
    """
    A LinearOperator corresponding to a shift matrix

    Attributes
    ----------
    shift: signed int
        Output vector elements are shifted according to the shift attribute.
        If positive, shift toward superdiagonals, and toward subdiagonal if negative.

    References
    ----------
    http://en.wikipedia.org/wiki/Shift_matrix
    """
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
    """
    Perform permutations to the vector elements.

    Attributes
    ----------
    perm : list or tuple
        The permutation of coefficients.
    """
    def __init__(self, perm, **kwargs):
        """
        Parameters
        ----------
        perm : list or tuple
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

        shape = 2 * (len(perm),)
        self.perm = perm
        # reverse permutation
        self.perm_inv = np.argsort(perm)
        def matvec(x):
            return np.asarray([x[pi] for pi in self.perm])
        def rmatvec(x):
            return np.asarray([x[pi] for pi in self.perm_inv])
        LinearOperator.__init__(self, shape, matvec, rmatvec=rmatvec, **kwargs)
    def __repr__(self):
        s = LinearOperator.__repr__(self)
        return s[:-1] + ",\n permutation=" + self.perm.__repr__() + ">"

class FftOperator(LinearOperator):
    """
    Generate an Operator performing Fast Fourier Transform.
    Uses rescaled numpy.fft.ftt and numpy.fft.ifft as matvec and rmatvec.
    """
    def __init__(self, shape, **kwargs):
        matvec = lambda x: np.fft.fft(x, n=shape[0]) / np.sqrt(shape[0])
        rmatvec = lambda x: np.fft.ifft(x, n=shape[1]) * np.sqrt(shape[0])
        LinearOperator.__init__(self, shape, matvec, rmatvec=rmatvec, **kwargs)

class ReplicationOperator(LinearOperator):
    """
    Generate an operator which replicates the input vector n times.
    """
    def __init__(self, shape, rep_num, **kwargs):
        """
        Parameters
        ----------
        shape : length 2 tuple.
            The shape of the operator.

        rep_num : int
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
        self.rep_num = rep_num
        # ensure coherent input
        if not shape[0] == shape[1] * rep_num:
            raise ValueError("Output vector should be n times the size of input vector.")
        def matvec(x):
            return np.tile(x, self.rep_num)
        def rmatvec(x):
            N = shape[1]
            return sum([x[i * N:(i + 1) * N] for i in xrange(self.rep_num)])
        LinearOperator.__init__(self, shape, matvec, rmatvec=rmatvec, **kwargs)
    def __repr__(self):
        s = LinearOperator.__repr__(self)
        return s[:-1] + ",\n Replicate %i times" % self.n + ">"

class SliceOperator(LinearOperator):
    """
    Perform slicing on the input vector.

    Attributes
    ----------
    slice: slice
       Slice applied to the input vector.
    """
    def __init__(self, shape, slice, **kwargs):
        """

        Exemple
        -------
        >>> S = lo.SliceOperator((2, 4), slice(None, None, 2))
        >>> S.todense()
        array([[ 1.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  0.]])

        >>> S.T.todense()
        array([[ 1.,  0.],
               [ 0.,  0.],
               [ 0.,  1.],
               [ 0.,  0.]])

        """
        def matvec(x):
            return x[slice]
        def rmatvec(x):
            out = np.zeros(shape[1])
            out[slice] = x
            return out
        LinearOperator.__init__(self, shape, matvec, rmatvec=rmatvec, **kwargs)

class TridiagonalOperator(LinearOperator):
    """
    Store a tridiagonal operator in the form of 3 arrays

    Attributes
    -----------
    shape : length 2 tuple.
        The shape of the operator.

    diag : ndarray of size shape[0]
        The diagonal of the matrix.

    subdiag : ndarray of size shape[0] - 1
        The subdiagonal of the matrix.

    superdiag : ndarray of size shape[0] - 1
        The superdiagonal of the matrix.
    """
    def __init__(self, shape, diag, subdiag, superdiag, **kwargs):
        """
        Parameters
        ----------
        shape : length 2 tuple.
            The shape of the operator.

        diag : ndarray of size shape[0]
            The diagonal of the matrix.

        subdiag : ndarray of size shape[0] - 1
            The subdiagonal of the matrix.

        superdiag : ndarray of size shape[0] - 1
            The superdiagonal of the matrix.

        Returns
        -------
        A Tridiagonal matrix operator instance.

        Exemple
        -------
        >>> import numpy as np
        >>> import linear_operators as lo
        >>> T = lo.TridiagonalOperator((3, 3), [1, 2, 3], [4, 5], [6, 7])
        >>> T.todense()
        array([[1, 6, 0],
               [4, 2, 7],
               [0, 5, 3]])
        """
        if shape[0] != shape[1]:
            raise ValueError("Only square operator is implemented.")
        self.diag = diag
        self.subdiag = subdiag
        self.superdiag = superdiag
        self.kwargs = kwargs

        def matvec(x):
            out = self.diag * x
            out[:-1] += self.superdiag * x[1:]
            out[1:] += self.subdiag * x[:-1]
            return out

        def rmatvec(x):
            out = self.diag * x
            out[:-1] += self.subdiag * x[1:]
            out[1:] += self.superdiag * x[:-1]
            return out

        LinearOperator.__init__(self, shape, matvec, rmatvec=rmatvec, **kwargs)

    def __repr__(self):
        s = LinearOperator.__repr__(self)[:-1]
        s += ",\n superdiagonal=" + self.superdiag.__repr__()
        s +=  ",\n diagonal=" + self.diag.__repr__()
        s += ",\n subdiagonal=" + self.subdiag.__repr__() + ">"
        return s

    def todense(self):
        out = np.zeros(self.shape, dtype=self.dtype)
        out += np.diag(self.diag)
        out += np.diag(self.subdiag, -1)
        out += np.diag(self.superdiag, 1)
        return out

    def __getitem__(self, y):
        # if tuple work on two dimensions
        if isinstance(y, tuple):
            # test dimension
            if len(y) > 2:
                raise IndexError("This is a 2-dimensional array.")
            yi, yj = y
            # single element case
            if isinstance(yi, int) and isinstance(yj, int):
                n = self.shape[0]
                i, j = yi % n , yj % n
                # outside
                if np.abs(i - j) > 1:
                    return self.dtype(0)
                # subdiag
                elif i == j + 1:
                    # border case
                    if i == self.shape[0] - 1:
                        return self.dtype(0)
                    else:
                        return self.subdiag[i]
                # superdiag
                elif i == j - 1:
                    # border case
                    if i == self.shape[0]:
                        return self.dtype(0)
                    else:
                        return self.superdiag[i]
                # diag
                else:
                    return self.diag[i]
            # case of tuple of length 1
            elif len(y) == 1:
                return self.__getitem__(self, y[0])
            # get a column
            elif yi == slice(None, None) and isinstance(yj, int):
                x = np.zeros(self.shape[1], dtype=self.dtype)
                x[yj] = 1.
                return self * x
            # general case: no better way than todense
            else:
                d = self.todense()
                return d[y]
        # Work on lines : same cost as recasting to a dense matrix as
        # all columns need to be accessed.
        else:
            d = self.todense()
            return d[y]

    @property
    def T(self):
        kwargs = dict()
        kwargs["dtype"] = getattr(self, 'dtype', None)
        kwargs["dtypein"] = getattr(self, 'dtypein', None)
        kwargs["dtypeout"] = getattr(self, 'dtypeout', None)
        return TridiagonalOperator(self.shape, self.diag, self.superdiag,
                                   self.subdiag, **kwargs)

    def toband(self):
        """
        Convert the TridiagonalOperator into a BandOperator
        """
        kl, ku = 1, 1
        n = self.shape[1]
        ab = np.zeros((kl + ku + 1, n))
        diags = (self.subdiag, self.diag, self.superdiag)
        for i, d in zip((-1, 0, 1), diags):
            ab[_band_diag(ku, i)] = d
        return BandOperator(self.shape, ab, kl, ku, **self.kwargs)

# Multiple inheritence
class SymmetricTridiagonal(SymmetricOperator, TridiagonalOperator):
    def __init__(self, shape, diag, subdiag, **kwargs):
        """
        Parameters
        ----------
        shape : length 2 tuple.
            The shape of the operator.

        diag : ndarray of size shape[0]
            The diagonal of the matrix.

        subdiag : ndarray of size shape[0] - 1
            The subdiagonal and superdiagonal of the matrix.

        Returns
        -------
        A Tridiagonal matrix operator instance.

        Exemple
        -------
        >>> import numpy as np
        >>> import linear_operators as lo
        >>> T = lo.TridiagonalOperator((3, 3), [1, 2, 3], [4, 5])
        >>> T.todense()
        array([[1, 4, 0],
               [4, 2, 5],
               [0, 5, 3]])
        """
        return TridiagonalOperator.__init__(self, shape, diag, subdiag,
                                            subdiag, **kwargs)

    def toband(self):
        """
        Convert into a SymmetricBandOperator.
        """
        u = 2 # tridiagonal
        n = self.shape[0]
        # convert to ab format (lower)
        ab = np.zeros((u, n))
        ab[0] = self.diag
        ab[1, :-1] = self.subdiag
        # options
        kwargs = dict()
        kwargs["dtype"] = getattr(self, 'dtype', None)
        kwargs["dtypein"] = getattr(self, 'dtypein', None)
        kwargs["dtypeout"] = getattr(self, 'dtypeout', None)
        return SymmetricBandOperator(self.shape, ab, lower=True, **kwargs)

class BandOperator(LinearOperator):
    """
    Store a band matrix in ab format as defined in LAPACK
    documentation.

    a[i, j] is stored in ab[ku + 1 + i - j, j]

    for max(1, j -ku) < i < min(m, j + kl)

    Band storage of A (5, 5), kl = 2, ku = 1 :

     *  a01 a12 a23 a34
    a00 a11 a22 a33 a44
    a10 a21 a32 a43  *
    a20 a31 a42  *   *

    Arguments
    ----------
    shape : 2-tuple
        Shape of the dense matrix equivalent.
    kl : int
        Number of subdiagonals
    ku : int
        Number of superdiagonals

    Notes
    -----
    For a description of band matrices see LAPACK doc :

    http://www.netlib.org/lapack/lug/node124.html

    """
    def __init__(self, shape, ab, kl, ku, **kwargs):
        """
        Generate a BandOperator instance

        Arguments
        ---------
        shape : 2-tuple
           The shape of the operator
        ab : ndarray with ndim == 2
           Store the bands of the matrix using LAPACK storage scheme.
        kl : int
            Number of subdiagonals
        ku : int
            Number of superdiagonals
        """
        if ab.shape[0] != kl + ku + 1 or ab.shape[1] != shape[1]:
            raise ValueError("Wrong ab shape.")

        self.ab = ab
        self.kl = kl
        self.ku = ku
        self.kwargs = kwargs

        def matvec(x):
            # diag
            out = self.ab[ku] * x
            # upper part
            for i in xrange(ku):
                j = ku - i
                out[:-j] += self.ab[i, j:] * x[j:]
            for i in xrange(ku, kl + ku):
            # lower part
                out[i:] += self.ab[i + 1, :-i] * x[:-i]
            return out

        def rmatvec(x):
            rab = self._rab
            rkl, rku = ku, kl
            # diag
            out = rab[ku] * x
            # upper part
            for i in xrange(rku):
                j = rku - i
                out[:-j] += rab[i, j:] * x[j:]
            for i in xrange(rku, rkl + rku):
            # lower part
                out[i:] += rab[i + 1, :-i] * x[:-i]
            return out

        return LinearOperator.__init__(self, shape, matvec, rmatvec,
                                       **kwargs)

    @property
    def T(self):
        return BandOperator(self.shape[::-1], self.rab, self.ku, self.kl,
                            **self.kwargs)

    def diag(self, i=0):
        """
        Returns the i-th diagonal (subdiagonal if i < 0, superdiagonal
        if i >0).
        """
        return self.ab[_band_diag(self.ku, i)]

    @property
    def rab(self):
        """
        Output the ab form of the transpose operator.
        """
        ab = self.ab
        kl, ku = self.kl, self.ku
        rku, rkl = kl, ku
        rab = np.zeros(ab.shape, dtype=ab.dtype)
        for i in xrange(- kl, ku + 1):
            rab[_band_diag(rku, -i)] = self.diag(i)
        return rab

def _band_diag(ku, i=0):
    """
    Return a slice to get the i-th line of a band operator
    """
    # diagonal
    if i == 0:
        return slice(ku, ku + 1)
    # superdiagonal
    if i > 0:
        return (slice(ku - i, ku - i + 1, None), slice(i, None, None))
    # subdiagonal
    if i < 0:
        return (slice(ku - i, ku - i + 1, None), slice(None, i, None))

class LowerTriangularOperator(BandOperator):
    def __init__(self, shape, ab, **kwargs):
        kl = ab.shape[0] - 1
        ku = 0
        BandOperator.__init__(self, shape, ab, kl, ku, **kwargs)

class UpperTriangularOperator(BandOperator):
    def __init__(self, shape, ab, **kwargs):
        kl = 0
        ku = ab.shape[0] - 1
        BandOperator.__init__(self, shape, ab, kl, ku, **kwargs)

class SymmetricBandOperator(SymmetricOperator):
    def __init__(self, shape, ab, lower=True, **kwargs):
        if shape[0] != ab.shape[1]:
            raise ValueError("ab.shape[1] should be equald to shape[0].")
        if lower is False:
            raise NotImplemented

        self.ab = ab
        self.lower = lower
        self.kwargs = kwargs

        def matvec(x):
            out = self.ab[0] * x
            for i in xrange(1, self.ab.shape[0]):
                # upper part
                out[:-i] += self.ab[i, :-i] * x[i:]
                # lower part
                out[i:] += self.ab[i, :-i] * x[:-i]
            return out

        return SymmetricOperator.__init__(self, shape, matvec, **kwargs)

    @property
    def rab(self):
        return self.ab

    def eigen(self, eigvals_only=False, overwrite_a_band=False, select='a',
              select_range=None, max_ev=0):
        """
        Solve real symmetric or complex hermitian band matrix
        eigenvalue problem.

        Uses scipy.linalg.eig_banded function.
        """
        from scipy.linalg import eig_banded

        w, v = eig_banded(self.ab, lower=self.lower,
                          eigvals_only=eigvals_only,
                          overwrite_a_band=overwrite_a_band,
                          select=select,
                          select_range=select_range,
                          max_ev=max_ev)
        return EigendecompositionOperator(w=w, v=v)

    def cholesky(self, overwrite_ab=False):
        """
        Chlesky decomposition.
        Operator needs to be positive-definite.

        Uses scipy.linalg.cholesky_banded.

        Returns a matrix in ab form
        """
        from scipy.linalg import cholesky_banded

        ab_chol = cholesky_banded(self.ab,
                               overwrite_ab=overwrite_ab,
                               lower=self.lower)
        if self.lower:
            out = LowerTriangularOperator(self.shape, ab_chol, **self.kwargs)
        else:
            out = UpperTriangularOperator(self.shape, ab_chol, **self.kwargs)
        return out

class EigendecompositionOperator(SymmetricOperator):
    """
    Define a SymmetricOperator from the eigendecomposition of another
    SymmetricOperator. This can be used as an approximation for the
    operator.

    Inputs
    -------

    A: LinearOperator (default: None)
      The LinearOperator to approximate.
    v: 2d ndarray (default: None)
      The eigenvectors as given by arpack.eigsh
    w: 1d ndarray (default: None)
      The eigenvalues as given by arpack.eigsh
    **kwargs: keyword arguments
      Passed to the arpack.eigsh function.

    You need to specify either A or v and w.

    Returns
    -------

    An EigendecompositionOperator instance, which is a subclass of the
    SymmetricOperator.

    Notes
    -----

    This is really a wrapper for
    scipy.sparse.linalg.eigen.arpack.eigsh
    """
    def __init__(self, A=None, v=None, w=None, **kwargs):
        from interface import aslinearoperator
        kwargs['return_eigenvectors'] = True
        if v is None or w is None:
            w, v = eigsh(A, **kwargs)
            shape = A.shape
            dtype = A.dtype
            dtypein = A.dtypein
            dtypeout = A.dtypeout
        else:
            shape = 2 * (v.shape[0],)
            dtype = v.dtype
            dtypein = dtypeout = dtype
        W = diagonal(w)
        V = aslinearoperator(v)
        M = V * W * V.T
        # store some information
        self.eigenvalues = w
        self.eigenvectors = v
        self.kwargs = kwargs
        SymmetricOperator.__init__(self, shape, M.matvec,
                                             dtypein=dtypein,
                                             dtypeout=dtypeout,
                                             dtype=dtype)
    def det(self):
        """
        Output an approximation of the determinant from the
        eigenvalues.
        """
        return np.prod(self.eigenvalues)

    def logdet(self):
        """
        Output the log of the determinant. Useful as the determinant
        of large matrices can exceed floating point capabilities.
        """
        return np.sum(np.log(self.eigenvalues))

    def __pow__(self, n):
        """
        Raising an eigendecomposition to an integer power requires
        only raising the eigenvalues to this power.
        """
        return EigendecompositionOperator(v=self.eigenvectors,
                                          w=self.eigenvalues ** n,
                                          kwargs=self.kwargs)

    def inv(self):
        """
        Returns the pseudo-inverse of the operator.
        """
        return self ** -1

    def trace(self):
        return np.sum(self.eigenvalues)

    def cond(self):
        """
        Output an approximation of the condition number by taking the
        ratio of the maximum over the minimum eigenvalues, removing
        the zeros.

        For better approximation of the condition number, one should
        consider generating the eigendecomposition with the keyword
        which='BE', in order to have a correct estimate of the small
        eigenvalues.
        """
        nze = self.eigenvalues[self.eigenvalues != 0]
        return nze.max() / nze.min()

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

def slice_operator(shape, slice, **kwargs):
    return SliceOperator(shape, slice, **kwargs)

def tridiagonal(shape, diag, subdiag, superdiag, **kwargs):
    return TridiagonalOperator(shape, diag, subdiag, superdiag, **kwargs)

def symmetric_tridiagonal(shape, diag, subdiag, **kwargs):
    return SymmetricTridiagonal(shape, diag, subdiag, **kwargs)

def band_operator(shape, ab, kl, ku, **kwargs):
    return BandOperator(shape, ab, kl, ku, **kwargs)

def symmetric_band_operator(shape, ab, lower=True, **kwargs):
    return SymmetricBandOperator(shape, ab, lower=lower, **kwargs)

# utilities

def band_approximation(mat, kl=0, ku=0):
    """
    Approximate a dense matrix as a BandOperator.
    """
    ab = np.zeros((kl + ku + 1, mat.shape[1]), dtype=mat.dtype)
    # diag
    ab[ku] = np.diag(mat)
    # upper
    for i in xrange(ku):
        j = ku - i
        ab[i, j:] = np.diag(mat, j)
    # lower
    for i in xrange(ku + 1, kl + ku + 1):
        j = ku - i
        ab[i, :j] = np.diag(mat, j)
    return BandOperator(mat.shape, ab, kl, ku, dtype=mat.dtype)

def symmetric_band_approximation(mat, k=0):
    """
    Approximate a symmetrix matrix as a SymmetricBandOperator.
    """
    ab = np.zeros((k + 1, mat.shape[1]), dtype=mat.dtype)
    ab[0] = np.diag(mat)
    for i in xrange(1, k + 1):
        ab[i, :-i] = np.diag(mat, i)
    return SymmetricBandOperator(mat.shape, ab, dtype=mat.dtype)
