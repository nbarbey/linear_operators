"""
Implement algorithms using the LinearOperator class
"""
import numpy as np
import scipy.sparse.linalg as spl
from copy import copy
import lo
import pywt
from pywt import thresholding

def rls(M, Ds, hypers, b, optimizer=spl.cgs, **kargs):
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
    callback = getattr(kargs, 'callback', None)
    if callback is None:
        kargs['callback'] = CallbackFactory(verbose=True)
    M = lo.aslinearoperator(M)
    X = M.T * M
    for h, D in zip(hypers, Ds):
        D = lo.aslinearoperator(D)
        X += h * D.T * D
    return optimizer(X, M.T * b, **kargs)

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

def rirls(M, Ds, y, tol1=1e-5, maxiter1=10, p=1, optimizer=spl.cgs, **kargs):
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

def gacg(M, Ds, hypers, norms, dnorms, y, tol=1e-6, x0=None, maxiter=None,
         callback=None):
    """Generalized approximate conjugate gradient
    
    Approximate conjugate gradient is a gradient method with a polak
    ribiere update.

    It is generalized since it does not assume quadratic norm.

    Inputs:
    -------
    M : model matrix, (needs matvec and rmatvec methods)
    Ds : priors, (need matvec and rmatvec methods)
    norms : norms of the likelihood and the priors
    dnorms : derivation of the norm
    b : input vector
    hypers: hyperparameters (scalars)


    Outputs:
    --------
    x : solution

    """
    from copy import copy
    if callback is None:
        callback = CallbackFactory(verbose=True)
    # ensure linear operators are passed
    M = lo.aslinearoperator(M)
    Ds = [lo.aslinearoperator(D) for D in Ds]
    # first guess
    if x0 is None:
        x = M.T * y
    else:
        x = copy(x0)
    # tolerance
    r = M * x - y
    J = norms[0](r)
    J += np.sum([h * norm(D * x) for norm, h, D in zip(norms[1:], hypers, Ds)])
    Jnorm = copy(J)
    resid = 2 * tol
    # maxiter
    if maxiter is None:
        maxiter = x.size
    iter_ = 0
    # main loop
    while iter_ < maxiter and resid > tol:
        iter_ += 1
        # gradient
        g = M.T * dnorms[0](r)
        ng = norm2(g)
        rd = []
        for h, D, dnorm in zip(hypers, Ds, dnorms[1:]):
            rd.append(D * x)
            g += h * D.T * dnorm(rd[-1])
        # descent direction
        if iter_ == 1:
            d = - g
        else:
            b = ng / ng0
            d = - g + b * d
        g0 = copy(g)
        ng0 = copy(ng)
        # step
        a = -.5 * np.dot(d.T, g)
        a /= norm2(M * d) + np.sum([h * norm2(D * d) for h, D in zip(hypers, Ds)])
        # update
        x += a * d
        # residual
        r = M * x - y
        # criterion
        J0 = copy(J)
        J = norms[0](r) 
        J += np.sum([h * norm(el) for norm, h, el in zip(norms[1:], hypers, rd)])
        resid = (J0 - J) / Jnorm
        callback(x)
        if resid > tol:
            info = resid
        else:
            info = 0
    return x, info

def acg(M, Ds, hypers, y, **kargs):
    "Approximate Conjugate gradient"
    norms = (norm2, ) * (len(Ds) + 1)
    dnorms = (dnorm2, ) * (len(Ds) + 1)
    return gacg(M, Ds, hypers, norms, dnorms, y, **kargs)

def hacg(M, Ds, hypers, y, deltas=None, **kargs):
    "Huber Approximate Conjugate gradient"
    if deltas is None:
        return acg(M, Ds, hypers, y, **kargs)
    norms = [hnorm(delta) for delta in deltas]
    dnorms = [dhnorm(delta) for delta in deltas]
    return gacg(M, Ds, hypers, norms, dnorms, y, **kargs)

def npacg(M, Ds, hypers, ps, y, **kargs):
    "Norm p Approximate Conjugate gradient"
    norms = [normp(p) for p in ps]
    dnorms = [dnormp(p) for p in ps]
    return gacg(M, Ds, hypers, norms, dnorms, y, **kargs)

# norms

def norm2(x):
    return np.dot(x.T, x)

def dnorm2(x):
    return 2 * x

def normp(p=2):
    def norm(t):
        return np.sum(np.abs(t) ** p)
    return norm

def dnormp(p=2):
    def norm(t):
        return np.sign(t) * p * (np.abs(t) ** (p - 1))
    return norm

def hnorm(d=None):
    if d is None:
        return norm2
    else:
        def norm(t):
            return np.sum(huber(t, d))
        return norm

def dhnorm(d=None):
    if d is None:
        return dnorm2
    else:
        def norm(t):
            return dhuber(t, d)
        return norm

def huber(t, delta=1):
    """Apply the huber function to the vector t, with transition delta"""
    t_out = t.flatten()
    quadratic_index = np.where(np.abs(t_out) < delta)
    linear_index = np.where(np.abs(t_out) >= delta)
    t_out[quadratic_index] = np.abs(t_out[quadratic_index]) ** 2
    t_out[linear_index] = 2 * delta * np.abs(t_out[linear_index]) - delta ** 2
    return np.reshape(t_out, t.shape)

def dhuber(t, delta=1):
    """Apply the derivation of the Huber function to t, transition: delta"""
    t_out = t.flatten()
    quadratic_index = np.where(np.abs(t_out) < delta)
    linear_index_positive = np.where(t_out >= delta)
    linear_index_negative = np.where(t_out <= - delta)
    t_out[quadratic_index] = 2 * t_out[quadratic_index]
    t_out[linear_index_positive] = 2 * delta
    t_out[linear_index_negative] = - 2 * delta
    return np.reshape(t_out, t.shape)

# Iterative Thresholding methods

def landweber(A, W, y, mu=1., nu=None, threshold=thresholding.hard, x0=None,
              maxiter=100, callback=None):
    """Landweber algorithm
    
    Input
    -----
    A : measurement matrix
    W : wavelet transform
    y : data
    mu : step of the gradient update
    nu : thresholding coefficient
    threshold : thresholding function
    maxiter : number of iterations
    callback : callback function
    
    Output
    ------
    
    x : solution
    
    """
    if callback is None:
        callback = lo.CallbackFactory(verbose=True)
    if x0 is None:
        x = A.T * y
    else:
        x = copy(x0)
    for iter_ in xrange(maxiter):
        r = A * x - y
        x += .5 * mu * A.T * (r)
        x = W.T * threshold(W * x, nu)
        resid = lo.norm2(r) + 1 / (2 * nu) * lo.normp(p=1)(x)
        callback(x)
    return x

def fista(A, W, y, mu=1., nu=None, threshold=thresholding.hard, x0=None,
          maxiter=100, callback=None):
    """ Fista algorithm
    
    """
    if callback is None:
        callback = lo.CallbackFactory(verbose=True)
    if x0 is None:
        x = A.T * y
    else:
        x = copy(x0)
    x_old = np.zeros(x.shape)
    t_old = 1.
    for iter_ in xrange(maxiter):
        t = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
        a = (t_old - 1) / t
        t_old = copy(t)
        z = x + a * (x - x_old)
        g = A.T * (A * z - y)
        x = z - .5 * mu * g
        x = W.T * threshold(W * x, nu)
        x_old = copy(x)
        resid = lo.norm2(A * x - y) + 1 / (2 * nu) * lo.normp(p=1)(x)
        callback(x)
    return x

# To create callback functions
class CallbackFactory():
    def __init__(self, verbose=False):
        self.iter_ = []
        self.resid = []
        self.verbose = verbose
    def __call__(self, x):
        import inspect
        parent_locals = inspect.stack()[1][0].f_locals
        self.iter_.append(parent_locals['iter_'])
        self.resid.append(parent_locals['resid'])
        if self.verbose:
            print('Iteration: ' + str(self.iter_[-1]) + '\t'
                  + 'Residual: ' + str( self.resid[-1]))

