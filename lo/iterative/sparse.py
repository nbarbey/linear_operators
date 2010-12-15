"""
Implement algorithms using the LinearOperator class
"""
import numpy as np
import scipy.sparse.linalg as spl
from copy import copy
import lo
from lo.iterative import *

def rls(A, b, Ds=[], hypers=[], optimizer=spl.cgs, **kargs):
    """Regularized Least Square
    
    Inputs:
    -------
    M : model matrix, (needs matvec and rmatvec methods)
    Dx : priors, (need matvec and rmatvec methods)
    b : input vector
    hypers: hyperparameters (scalars)
    **kargs : parameters of of the least square optimizer

    Outputs:
    --------
    x : solution
    conv : convergence status

    """
    verbose = getattr(kargs, 'verbose', True)
    callback = getattr(kargs, 'callback', None)
    # normalize hyperparameters
    if callback is None:
        kargs['callback'] = CallbackFactory(verbose=verbose)
    A = lo.aslinearoperator(A)
    X = A.T * A
    for h, D in zip(hypers, Ds):
        D = lo.aslinearoperator(D)
        X += h * D.T * D
    out, info = optimizer(X, A.T * b, **kargs)
    return out

def irls(A0, x0, tol1=1e-5, maxiter1=10, p=1, optimizer=spl.cgs, **kargs):
    """ Iteratively Reweighted Least Square
        
    """
    A0 = lo.aslinearoperator(A0)
    out = A0.T * x0
    tol1 /= np.sum(np.abs(x0 - A0 * out) ** p)
    for i in xrange(maxiter1):
        print("\n outer loop " + str(i + 1) + "\n")
        w = np.abs(x0 - A0 * out) ** (p - 2)
        A = A0.T * lo.diag(w) * A0
        x = A0.T * (w * x0)
        out, t = optimizer(A, x, **kargs)
        if np.sum(np.abs(x0 - A0 * out) ** p) < tol1:
            break
    return out

def rirls(M, y, Ds=[], tol1=1e-5, maxiter1=10, p=1, optimizer=spl.cgs, **kargs):
    """ Regularized Iteratively Reweighted Least Square
        
    """
    M = lo.aslinearoperator(M)
    Ds = [lo.aslinearoperator(D) for D in Ds]
    x0 = M.T * y
    x = copy(x0)
    r = M * x - y
    tol1 /= np.sum(np.abs(r) ** p)
    for i in xrange(maxiter1):
        print("\n outer loop " + str(i + 1) + "\n")
        A = M.T * M
        for D in Ds:
            rd = D * x
            w = np.abs(rd) ** (p - 2)
            w[np.where(1 - np.isfinite(w))] = 0 # inf
            A += D.T * lo.diag(w) * D
        x, t = optimizer(A, x0, **kargs)
        if np.sum(np.abs(y - M * x) ** p) < tol1:
            break
        r = M * x - y
    return x

