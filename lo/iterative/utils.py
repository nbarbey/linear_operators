"""
Usefull functions for LinearOperators
"""
import numpy as np
try:
    from scipy.sparse.linalg import arpack
except(ImportError):
    import arpack

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
            print "All small eigenvalues were zeros."
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
