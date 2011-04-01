"""
Usefull functions for LinearOperators
"""
import numpy as np
try:
    from scipy.sparse.linalg import arpack
except(ImportError):
    import arpack

from .. import operators

def eigen_symmetric(A, **kwargs):
    """
    A wrapper to arpack eigen_symmetric which output an approximation
    of the A input matrix, as a LinearOperator storing eigenvalues and
    eigenvectors.

    It passes its arguments arguments as arpack.eigen_symmetric but
    forces return_eigenvectors to True.
    """
    return EigendecompositionOperator(A, **kwargs)

class EigendecompositionOperator(operators.SymmetricOperator):
    """
    Define a SymmetricOperator from the eigendecomposition of another
    SymmetricOperator. This can be used as an approximation for the
    operator.
    """
    def __init__(self, A, **kwargs):
        from ..interface import aslinearoperator
        from ..operators import diagonal
        kwargs['return_eigenvectors'] = True
        w, v = arpack.eigen_symmetric(A, **kwargs)
        W = diagonal(w)
        V = aslinearoperator(v)
        M = V * W * V.T
        # store some information
        self.eigenvalues = w
        self.eigenvectors = v
        self.k = kwargs['k']
        operators.SymmetricOperator.__init__(self, A.shape, M.matvec,
                                             dtypein=A.dtypein,
                                             dtypeout=A.dtypeout,
                                             dtype=A.dtype)
    def det(self):
        """
        Output an approximation of the determinant from the
        eigenvalues.
        """
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
        eigen = arpack.eigen_symmetric
    else:
        eigen = arpack.eigen

    vmax = eigen(A, which='LM', k=kl, M=M, maxiter=maxiter, tol=tol, return_eigenvectors=False)
    vmin = eigen(A, which='SM', k=ks, M=M, maxiter=maxiter, tol=tol, return_eigenvectors=False)
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
