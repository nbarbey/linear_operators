"""
Implement algorithms using the LinearOperator class
"""
import numpy as np
import scipy.sparse.linalg as spl
from copy import copy
import lo
import pywt
from pywt import thresholding

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
    if callback is None:
        kargs['callback'] = CallbackFactory(verbose=verbose)
    A = lo.aslinearoperator(A)
    X = A.T * A
    for h, D in zip(hypers, Ds):
        D = lo.aslinearoperator(D)
        X += h * D.T * D
    return optimizer(X, A.T * b, **kargs)

class Model():
    def __init__(self, M, b, Ds=[], hypers=[]):
        M = lo.aslinearoperator(M)
        Ds = [lo.aslinearoperator(D) for D in Ds]
        self.M = M
        self.b = b
        self.Ds = Ds
        self.hypers = hypers
    def __call__(self, x):
        out = norm2(self.b - self.M * x) 
        out += np.sum([h * norm2(D * x) 
                       for D, h in zip(self.Ds, self.hypers)])
        return out
    def gradient(self, x):
        r = self.M * x - self.b
        out = self.M.T * dnorm2(r)
        out += np.sum([h * D.T * dnorm(D * x)
                       for D, h in zip(self.Ds, self.hypers)])
        return out

def opt(M, b, Ds=[], hypers=[], maxiter=None, tol=1e-6, min_alpha_step=0.0001):
    """
    Use scikits.optimization to perform least-square inversion.
    """
    from scikits.optimization import step, line_search, criterion, optimizer

    if maxiter is None:
        maxiter = M.shape[0]
    model = Model(M, b, Ds, hypers)
    mystep = step.GradientStep()
    mylinesearch = line_search.GoldenSectionSearch(min_alpha_step=min_alpha_step)
    mycriterion = criterion.criterion(ftol=tol, iterations_max=maxiter)
    myoptimizer = optimizer.StandardOptimizer(function=model, 
                                              step=mystep,
                                              line_search=mylinesearch,
                                              criterion=mycriterion,
                                              x0 = np.zeros(M.shape[1]))
    return myoptimizer.optimize()

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

def gacg(M, y, Ds=[], hypers=[], norms=[], dnorms=[], tol=1e-6, x0=None, maxiter=None,
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
    rd = [D * x for D in Ds]
    J = criterion(hypers=hypers, norms=norms, Ds=Ds, r=r, rd=rd)
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
        g, ng = gradient(hypers=hypers, dnorms=dnorms, M=M, Ds=Ds, r=r, rd=rd)
        # descent direction
        if iter_ == 1:
            d = - g
        else:
            b = ng / ng0
            d = - g + b * d
        g0 = copy(g)
        ng0 = copy(ng)
        # step
        if norms[0] == norm2:
            a = quadratic_optimal_step(d, g, M, hypers, Ds)
        else:
            a = backtracking_line_search(d, g, M, hypers, Ds,
                                         x, norms)
        # update
        x += a * d
        # residual
        r = M * x - y
        rd = [D * x for D in Ds]
        # criterion
        J0 = copy(J)
        J = criterion(hypers=hypers, norms=norms, Ds=Ds, r=r, rd=rd)
        resid = J / Jnorm
        callback(x)
    # define output
    if resid > tol:
        info = resid
    else:
        info = 0
    return x, info

def criterion(x=None, y=None, M=None, norms=None, hypers=None, Ds=None,
              r=None, rd=None):
    if r is None:
        r = M * x - y
    if rd is None:
        rd = [D * x for D in Ds]
    J = norms[0](r)
    J += np.sum([h * norm(el) for norm, h, el in zip(norms[1:], hypers, rd)])
    return J

def gradient(x=None, y=None, M=None, dnorms=None, hypers=None, Ds=None,
             r=None, rd=None):
    if r is None:
        r = M * x - y
    g = M.T * dnorms[0](r)
    ng = norm2(g)
    if rd is None:
        rd = [D * x for D in Ds]
    g += np.sum([h * D.T * dnorm(el) for dnorm, h, D, el 
                 in zip(dnorms[1:], hypers, Ds, rd)])
    return g, ng

def acg(M, y, Ds=[], hypers=[], **kargs):
    "Approximate Conjugate gradient"
    norms = (norm2, ) * (len(Ds) + 1)
    dnorms = (dnorm2, ) * (len(Ds) + 1)
    return gacg(M, y, Ds, hypers, norms, dnorms, **kargs)

def hacg(M, y, Ds=[], hypers=[], deltas=None, **kargs):
    "Huber Approximate Conjugate gradient"
    if deltas is None:
        return acg(M, Ds, hypers, y, **kargs)
    norms = [hnorm(delta) for delta in deltas]
    dnorms = [dhnorm(delta) for delta in deltas]
    return gacg(M, y, Ds, hypers, norms, dnorms, **kargs)

def npacg(M, y, Ds=[], hypers=[], ps=[], **kargs):
    "Norm p Approximate Conjugate gradient"
    norms = [normp(p) for p in ps]
    dnorms = [dnormp(p) for p in ps]
    return gacg(M, y, Ds, hypers, norms, dnorms, **kargs)

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

# line search methods

def quadratic_optimal_step(d, g, M, hypers, Ds):
    a = -.5 * np.dot(d.T, g)
    a /= norm2(M * d) + np.sum([h * norm2(D * d) for h, D in zip(hypers, Ds)])
    return a

def backtracking_line_search(d, g, M, hypers, Ds, 
                             x0=None, norms=None, y=None, f0=None, maxiter=10, tau=.5):
    a = quadratic_optimal_step(d, g, M, hypers, Ds)
    i = 0
    fi = 2 * f0
    # XXX replace with armijo wolfe conditions
    while (i < maxiter) and (fi > f0):
        i += 1
        a *= tau
        xi = x + a * d
        fi = criterion(x=xi, y=y, M=M, hypers=hypers, norms=norms, Ds=Ds)
    return a

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
