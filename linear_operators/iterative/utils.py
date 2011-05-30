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
    from ..operators import EigendecompositionOperator
    return EigendecompositionOperator(A, **kwargs)

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
