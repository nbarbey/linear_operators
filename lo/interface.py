import numpy as np
#from scipy.sparse.sputils import isshape
#from scipy.sparse import isspmatrix

#__all__ = ['LinearOperator', 'aslinearoperator']

class LinearOperator:
    """Common interface for performing matrix vector products

    Many iterative methods (e.g. cg, gmres) do not need to know the
    individual entries of a matrix to solve a linear system A*x=b.
    Such solvers only require the computation of matrix vector
    products, A*v where v is a dense vector.  This class serves as
    an abstract interface between iterative solvers and matrix-like
    objects.

    Parameters
    ----------
    shape : tuple
        Matrix dimensions (M,N)
    matvec : callable f(v)
        Returns returns A * v.

    Optional Parameters
    -------------------
    rmatvec : callable f(v)
        Returns A^H * v, where A^H is the conjugate transpose of A.
    matmat : callable f(V)
        Returns A * V, where V is a dense matrix with dimensions (N,K).
    rmatmat : callable f(V)
        Returns A^H * V, where V is a dense matrix with dimensions (N,K).
    dtype : dtype
        Data type of the matrix.

    See Also
    --------
    aslinearoperator : Construct LinearOperators

    Notes
    -----
    The user-defined matvec() function must properly handle the case
    where v has shape (N,) as well as the (N,1) case.  The shape of
    the return type is handled internally by LinearOperator.

    Examples
    --------
    >>> from scipy.sparse.linalg import LinearOperator
    >>> from scipy import *
    >>> def mv(v):
    ...     return array([ 2*v[0], 3*v[1]])
    ...
    >>> A = LinearOperator( (2,2), matvec=mv, rmatvec=mv)
    >>> A
    <2x2 LinearOperator with unspecified dtype>
    >>> A.matvec( ones(2) )
    array([ 2.,  3.])
    >>> A * ones(2)
    array([ 2.,  3.])
    >>> A.dense()
    array([[ 2.,  0.],
           [ 0.,  3.]])
    >>> (2 * A.T * A + 1) * ones(2)
    array([  9.,  19.])

    """
    def __init__(self, shape, matvec, rmatvec=None, matmat=None, rmatmat=None,
                 dtypein=None, dtypeout=None, dtype=None):

        shape = tuple(shape)

        #if not isshape(shape):
        #    raise ValueError('invalid shape')

        self.shape  = shape
        self._matvec = matvec

        if rmatvec is None:
            def rmatvec(v):
                raise NotImplementedError('rmatvec is not defined')
            self.rmatvec = rmatvec
        else:
            self.rmatvec = rmatvec

        if matmat is not None:
            # matvec each column of V
            self._matmat = matmat

        if rmatmat is not None:
            # matvec each column of V
            self._rmatmat = rmatmat
        else:
            self._rmatmat = None

        self.dtype = None
        self.dtypein = None
        self.dtypeout = None
        if dtype is not None:
            self.dtype = dtype
            self.dtypein = np.dtype(dtype)
            self.dtypeout = np.dtype(dtype)
        if dtypein is not None and dtypeout is not None:
            self.dtype = np.dtype(dtypein)
            self.dtypein = np.dtype(dtypein)
            self.dtypeout = np.dtype(dtypeout)
        elif dtypein is not None:
            self.dtype = np.dtype(dtypein)
            self.dtypein = np.dtype(dtypein)
        elif dtypeout is not None:
            self.dtype = np.dtype(dtypeout)
            self.dtypeout = np.dtype(dtypeout)

    def _matmat(self, X):
        """Default matrix-matrix multiplication handler.  Falls back on
        the user-defined matvec() routine, which is always provided.
        """

        return np.hstack( [ self.matvec(col.reshape(-1,1)) for col in X.T ] )


    def matvec(self, x):
        """Matrix-vector multiplication

        Performs the operation y=A*x where A is an MxN linear
        operator and x is a column vector or rank-1 array.

        Parameters
        ----------
        x : {matrix, ndarray}
            An array with shape (N,) or (N,1).

        Returns
        -------
        y : {matrix, ndarray}
            A matrix or ndarray with shape (M,) or (M,1) depending
            on the type and shape of the x argument.

        Notes
        -----
        This matvec wraps the user-specified matvec routine to ensure that
        y has the correct shape and type.

        """

        x = np.asanyarray(x)

        M,N = self.shape

        if x.shape != (N,) and x.shape != (N,1):
            raise ValueError('dimension mismatch')

        y = self._matvec(x)

        if isinstance(x, np.matrix):
            y = np.asmatrix(y)
        else:
            y = np.asarray(y)

        if x.ndim == 1:
            y = y.reshape(M)
        elif x.ndim == 2:
            y = y.reshape(M,1)
        else:
            raise ValueError('invalid shape returned by user-defined matvec()')


        return y


    def matmat(self, X):
        """Matrix-matrix multiplication

        Performs the operation y=A*X where A is an MxN linear
        operator and X dense N*K matrix or ndarray.

        Parameters
        ----------
        X : {matrix, ndarray}
            An array with shape (N,K).

        Returns
        -------
        Y : {matrix, ndarray}
            A matrix or ndarray with shape (M,K) depending on
            the type of the X argument.

        Notes
        -----
        This matmat wraps any user-specified matmat routine to ensure that
        y has the correct type.

        """

        X = np.asanyarray(X)

        if X.ndim != 2:
            raise ValueError('expected rank-2 ndarray or matrix')

        M,N = self.shape

        if X.shape[0] != N:
            raise ValueError('dimension mismatch')

        Y = self._matmat(X)

        if isinstance(Y, np.matrix):
            Y = np.asmatrix(Y)

        return Y


    def __mul__(self, x):
        if np.isscalar(x):
            matvec = _mat_scalar(self._matvec, x)
            rmatvec = _mat_scalar(self.rmatvec, x)
            matmat = _mat_scalar(self._matmat, x)
            rmatmat = _mat_scalar(self._rmatmat, x)
            return LinearOperator(self.shape, matvec, rmatvec=rmatvec,
                                  matmat=matmat, rmatmat=rmatmat)
        if isinstance(x, LinearOperator):
            # return a LinearOperator
            if self.shape[1] != x.shape[0]:
                raise ValueError('LinearOperator shape mismatch')
            if self.dtypein != x.dtypeout:
                raise ValueError('LinearOperator dtypein mismatch')
            shape = (self.shape[0], x.shape[1])
            matvec = _mat_mul(self._matvec, x._matvec)
            if self.rmatvec is not None and x.rmatvec is not None:
                rmatvec = _mat_mul(x.rmatvec, self.rmatvec)
            else:
                rmatvec = None
            if self._matmat is not None and x._matmat is not None:
                matmat = _mat_mul(self._matmat, x._matmat)
            else:
                matmat = None
            if self._rmatmat is not None and x._matmat is not None:
                rmatmat = _mat_mul(x._rmatmat, self._rmatmat)
            else:
                rmatmat = None
            return LinearOperator(shape, matvec, rmatvec=rmatvec,
                                  matmat=matmat, rmatmat=rmatmat,
                                  dtypein=x.dtypein,
                                  dtypeout=self.dtypeout)
        else:
            x = np.asarray(x)
            if x.ndim == 1 or x.ndim == 2 and x.shape[1] == 1:
                return self.matvec(x).astype(self.dtypeout)
            elif x.ndim == 2:
                return self.matmat(x).astype(self.dtypeout)
            else:
                raise ValueError('expected rank-1 or rank-2 array or matrix or LinearOperator')

    def __add__(self, A):
        if isinstance(A, LinearOperator):
            if self.shape != A.shape:
                raise ValueError('expected LinearOperator of the same shape')
            if self.dtype != A.dtype:
                raise ValueError('LinearOperator dtype mismatch')
            if self.dtypein != A.dtypein:
                raise ValueError('LinearOperator dtypein mismatch')
            if self.dtypeout != A.dtypeout:
                raise ValueError('LinearOperator dtypeout mismatch')
            matvec = _mat_add(self._matvec, A._matvec)
            if self.rmatvec is not None and A.rmatvec is not None:
                rmatvec = _mat_add(self.rmatvec, A.rmatvec)
            else:
                rmatvec = None
            if self._matmat is not None and A._matmat is not None:
                matmat = _mat_add(self._matmat, A._matmat)
            else:
                matmat = None
            if self._matmat is not None and A._matmat is not None:
                rmatmat = _mat_add(self._matmat, A._matmat)
            else:
                rmatmat = None
            return LinearOperator(self.shape, matvec, rmatvec=rmatvec,
                                  matmat=matmat, rmatmat=rmatmat,
                                  dtype=self.dtype, dtypein=self.dtypein,
                                  dtypeout=self.dtypeout)
        if np.isscalar(A):
            return self.__add__(A * aslinearoperator(np.eye(self.shape[0],
                                                            self.shape[1],
                                                            dtype=self.dtype,
                                                            )))
        else:
            raise ValueError('expected LinearOperator')

    def __neg__(self):
        return self * (-1)

    def __sub__(self, x):
        return self.__add__(-x)

    def __rmul__(self, x):
        if np.isscalar(x):
            # commutative with scalar
            return self.__mul__(x)
        if isinstance(x, LinearOperator):
            return x.__mul__(self)
        else:
            x = np.asarray(x)
            if hasattr(self, 'rmatvec'):
                if x.ndim == 1 or x.ndim == 2 and x.shape[1] == 1:
                    return self.rmatvec(x)
                elif x.ndim == 2:
                    return self.rmatmat(x)
                else:
                    raise ValueError('expected rank-1 or rank-2 array or matrix')
            else:
                raise ValueError('LinearOperator does not have rmatvec attribute')

    def __radd__(self, x):
        return self.__add__(x)

    def __rsub__(self, x):
        return (-self).__add__(x)

    def __imul__(self, x):
        return self.__mul__(x)

    def __iadd__(self, x):
        return self.__add__(x)

    def __isub__(self, x):
        return self.__sub__(x)

    def __pow__(self, k):
        from copy import copy
        if not isinstance(k, int):
            raise ValueError('Only power to an int is implemented')
        if k < 0:
            raise ValueError('Negative power is not implemented')
        else:
            A = copy(self)
            for i in xrange(k):
                A *= self
            return A

    def __repr__(self):
        M,N = self.shape
        if hasattr(self,'dtype'):
            dt = 'dtype=' + str(self.dtypein)
        else:
            dt = 'unspecified dtype'

        return '<%dx%d LinearOperator with %s>' % (M,N,dt)

    @property
    def T(self):
        if self.rmatvec is not None:
            matvec, rmatvec = self.rmatvec, self._matvec
        else:
            raise NotImplementedError('rmatvec is not defined')
        if self._matmat is not None and self._rmatmat is not None:
            matmat, rmatmat = self._rmatmat, self._matmat
        else:
            matmat, rmatmat = None, None
        dtype = getattr(self, 'dtype', None)
        dtypein = getattr(self, 'dtypein', None)
        dtypeout = getattr(self, 'dtypeout', None)
        return LinearOperator(self.shape[::-1], matvec, rmatvec=rmatvec,
                              matmat=matmat, rmatmat=rmatmat, dtype=dtype,
                              dtypein=dtypeout, dtypeout=dtypein)

    def todense(self):
        A = np.empty(self.shape, dtype=self.dtype)
        x = np.zeros(self.shape[1], dtype=self.dtype)
        for i in xrange(A.shape[1]):
            x[i] = 1
            A[:, i] = self * x
            x[:] = 0
        return A

def _mat_mul(matvec1, matvec2):
    """Functional composition of two matvec functions"""
    def matvec(x):
        return matvec1(matvec2(x))
    return matvec

def _mat_add(matvec1, matvec2):
    """Functional addition of two matvec functions"""
    def matvec(x):
        return np.squeeze(matvec1(x)) + np.squeeze(matvec2(x))
    return matvec

def _mat_scalar(matvec0, scalar):
    def matvec(x):
        return scalar * matvec0(x)
    return matvec

def aslinearoperator(A):
    """Return A as a LinearOperator.

    'A' may be any of the following types:
     - ndarray
     - matrix
     - sparse matrix (e.g. csr_matrix, lil_matrix, etc.)
     - LinearOperator
     - An object with .shape and .matvec attributes

    See the LinearOperator documentation for additonal information.

    Examples
    --------
    >>> from scipy import matrix
    >>> M = matrix( [[1,2,3],[4,5,6]], dtype='int32' )
    >>> aslinearoperator( M )
    <2x3 LinearOperator with dtype=int32>

    """
    if isinstance(A, LinearOperator):
        return A

    elif isinstance(A, np.ndarray) or isinstance(A, np.matrix):
        if A.ndim > 2:
            raise ValueError('array must have rank <= 2')

        A = np.atleast_2d(np.asarray(A))

        def matvec(v):
            return np.dot(A, v)
        def rmatvec(v):
            return np.dot(A.conj().transpose(), v)
        def matmat(V):
            return np.dot(A, V)
        def rmatmat(V):
            return np.dot(V, A)
        return LinearOperator(A.shape, matvec, rmatvec=rmatvec,
                              matmat=matmat, rmatmat=rmatmat, dtype=A.dtype)

    #elif isspmatrix(A):
    #    def matvec(v):
    #        return A * v
    #    def rmatvec(v):
    #        return A.conj().transpose() * v
    #    def matmat(V):
    #        return A * V
    #    def rmatmat(V):
    #        return V * A
    #    return LinearOperator(A.shape, matvec, rmatvec=rmatvec,
    #                          matmat=matmat, rmatmat=rmatmat, dtype=A.dtype)

    else:
        if hasattr(A, 'shape') and hasattr(A, 'matvec'):
            rmatvec = getattr(A, 'rmatvec', None)
            matmat = getattr(A, 'matmat', None)
            rmatmat = getattr(A, 'rmatmat', None)
            dtype = getattr(A, 'dtype', None)
            return LinearOperator(A.shape, A.matvec, rmatvec=rmatvec,
                                  matmat=matmat, rmatmat=rmatmat, dtype=dtype)
        else:
            raise TypeError('type not understood')

def concatenate(As, axis=0):
    """
    Concatenate LinearOperator in rows or in columns.

    Parameters
    ----------
    As : list of LinearOperators
         The list of LinearOperators to concatenate.

    axis : 0 or 1
           The axis along which to concatenate the LinearOperators.

    Returns
    -------
    out: LinearOperator
      Output a LinearOperator which is the concatenation of a list of 
      LinearOpeartors.

    """
    # rows
    if axis == 0:
        # check input
        for A in As:
            # shape
            if A.shape[1] != As[0].shape[1]:
                raise ValueError("All LinearOperators must have the same number of row/columns.")
            # dtype
            if A.dtype != As[0].dtype:
                raise ValueError("All LinearOperators must have the same data-type.")
        # define output shape
        sizes = [A.shape[0] for A in As]
        shape = np.sum(sizes), As[0].shape[1]
        # define data type
        dtype = As[0].dtype
        # define new matvec
        def matvec(x):
            return np.concatenate([A.matvec(x) for A in As])
        # define how to slice vector
        sizesum = list(np.cumsum(sizes))[:-1]
        sizes1 = [None,] + sizesum
        sizes2 = sizesum + [None,]
        slices = [slice(s1, s2, None) for s1, s2 in zip(sizes1, sizes2)]
        def rmatvec(x):
            out = np.zeros(shape[1])
            for A, s in zip(As, slices):
                out += A.rmatvec(x[s])
            return out
    # columns
    elif axis == 1:
        # you can transpose the concatenation of the list of transposes
        return concatenate([A.T for A in As], axis=0).T
    else:
        raise ValueError("axis must be 0 or 1")
    return LinearOperator(shape, matvec, rmatvec=rmatvec, dtype=dtype)

def block_diagonal(As):
    """
    Defines a block diagonal LinearOperator from a list of LinearOperators.
    """
    # check data type
    for A in As:
        if A.dtype != As[0].dtype:
            raise ValueError("All LinearOperators must have the same data-type.")
    dtype = As[0].dtype
    # define shape
    shape = np.sum([A.shape[0] for A in As]), np.sum([A.shape[1] for A in As])
    # define how to slice vector
    sizes = [A.shape[1] for A in As]
    sizes = list(np.cumsum(sizes))[:-1]
    sizes1 = [None,] + sizes
    sizes2 = sizes + [None,]
    slices = [slice(s1, s2, None) for s1, s2 in zip(sizes1, sizes2)]
    # define new matvec function
    def matvec(x):
        return np.concatenate([A.matvec(x[s]) for A, s in zip(As, slices)])
    # define how to slice vector
    rsizes = [A.shape[0] for A in As]
    rsizes = list(np.cumsum(rsizes))[:-1]
    rsizes1 = [None,] + rsizes
    rsizes2 = rsizes + [None,]
    rslices = [slice(s1, s2, None) for s1, s2 in zip(rsizes1, rsizes2)]
    # define new rmatvec function
    def rmatvec(x):
        return np.concatenate([A.rmatvec(x[s]) for A, s in zip(As, rslices)])
    return LinearOperator(shape, matvec, rmatvec=rmatvec, dtype=dtype)
