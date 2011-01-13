"""
Usefull functions for LinearOperators
"""
import numpy as np

def cond(A, k=1, symmetric=True, M=None, maxiter=None, tol=0):
    """
    Find the condition number of a LinearOperator using arpack eigen
    function.
    """
    if symmetric:
        from scipy.sparse.linalg.arpack import eigen_symmetric as eigen
    else:
        from scipy.sparse.linalg.arpack import eigen

    smax = np.max(eigen(A, which='LM', k=k, M=M, maxiter=maxiter, tol=tol, return_eigenvectors=False))
    smin = np.min(eigen(A, which='SA', k=k, M=M, maxiter=maxiter, tol=tol, return_eigenvectors=False))
    
    return smax / smin
