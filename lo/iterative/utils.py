"""
Usefull functions for LinearOperators
"""
import numpy as np
try:
    from scipy.sparse.linalg import arpack
except(ImportError):
    import arpack

def cond(A, k=6, symmetric=True, M=None, maxiter=None, tol=1e-6, verbose=False):
    """
    Find the condition number of a LinearOperator using arpack eigen
    function.
    """
    if symmetric:
        from arpack import eigen_symmetric as eigen
    else:
        from arpack import eigen

    vmax = eigen(A, which='LM', k=k, M=M, maxiter=maxiter, tol=tol, return_eigenvectors=False)
    smax = np.max(vmax)
    if verbose:
        print vmax
        print smax
    
    vmin = eigen(A, which='SM', k=k, M=M, maxiter=maxiter, tol=tol, return_eigenvectors=False)
    # remove zeros
    vmin = vmin[vmin != 0.]
    smin = np.min(vmin)
    if verbose:
        print vmin
        print smin
    
    return smax / smin
