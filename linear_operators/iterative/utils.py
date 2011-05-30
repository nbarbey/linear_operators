"""
Useful functions for LinearOperators
"""
import numpy as np
try:
    from scipy.sparse.linalg import arpack
except(ImportError):
    import arpack

# to handle new names of eigen and eigen_symmetric
if 'eigen' in arpack.__dict__.keys():
    eigs = arpack.eigen
    eigsh = arpack.eigen_symmetric
elif 'eigs' in arpack.__dict__.keys():
    eigs = arpack.eigs
    eigsh = arpack.eigsh
else:
    raise ImportError("Unable to find eigen or eigs in arpack.")

from ..operators import SymmetricOperator

def eigendecomposition(A, **kwargs):
    """
    A wrapper to arpack eigen_symmetric which output an approximation
    of the A input matrix, as a LinearOperator storing eigenvalues and
    eigenvectors.

    It passes its arguments arguments as arpack.eigsh but
    forces return_eigenvectors to True.
    """
    return EigendecompositionOperator(A, **kwargs)

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
        from ..interface import aslinearoperator
        from ..operators import diagonal
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

def cond(A, k=2, kl=None, ks=None, symmetric=True, M=None, maxiter=None,
         tol=1e-6, verbose=False, prune_zeros=False, loweig=None):
    """
    Find the condition number of a LinearOperator using arpack eigen
    function.
    """
    if kl is None:
        kl = k
    if ks is None:
        ks = k

    if symmetric:
        eigen = eigsh
    else:
        eigen = eigs

    vmax = eigen(A, which='LM', k=kl, M=M, maxiter=maxiter, tol=tol,
                 return_eigenvectors=False)
    vmin = eigen(A, which='SM', k=ks, M=M, maxiter=maxiter, tol=tol,
                 return_eigenvectors=False)
    nb_null = np.sum(vmin == 0)
    if verbose:
        print("Found %d zero-valued eigenvalues" % nb_null)
        if nb_null == len(vmin):
            print("All small eigenvalues were zeros.")
    if prune_zeros:
        vmin = vmin[vmin !=0]
    if vmin.size == 0:
        return np.inf
    smin = np.min(vmin)
    smax = np.max(vmax)
    if smin == 0:
        return np.inf
    if loweig is not None:
        if smin < loweig:
            return np.inf
    if verbose:
        print vmin, vmax
        print smin, smax

    return smax / smin
