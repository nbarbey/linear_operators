"""
Implement algorithms using the LinearOperator class
"""
import numpy as np
from copy import copy
import lo

def gacg(M, y, Ds=[], hypers=[], norms=[], dnorms=[], C=None, tol=1e-6, x0=None, maxiter=None,
         callback=None, save=True, **kwargs):
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
    y : input vector
    hypers: hyperparameters (scalars)
    C : inverse of the covariance matrix

    Outputs:
    --------
    x : solution

    """
    if callback is None:
        callback = CallbackFactory(verbose=True, criterion=True,
                                   solution=True, save=save)
    # ensure linear operators are passed
    M = lo.aslinearoperator(M)
    Ds = [lo.aslinearoperator(D) for D in Ds]
    # first guess
    if x0 is None:
        if C is None:
            x = M.T * y
        else:
            x = M.T * C * y
    else:
        x = copy(x0)
    # normalize hyperparameters
    hypers = normalize_hyper(hypers, y, x)
    # tolerance
    r = M * x - y
    rd = [D * x for D in Ds]
    J = criterion(hypers=hypers, norms=norms, Ds=Ds, r=r, rd=rd, C=C)
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
        g, ng = gradient(hypers=hypers, dnorms=dnorms, M=M, Ds=Ds, r=r, rd=rd, C=C)
        # descent direction
        if (iter_  % 10) == 1:
            d = - g
        else:
            b = ng / ng0
            d = - g + b * d
        g0 = copy(g)
        ng0 = copy(ng)
        # step
        if norms[0] == norm2:
            a = quadratic_optimal_step(d, g, M, hypers, Ds, C=C)
        else:
            a = backtracking_line_search(d, g, M, hypers, Ds,
                                         x, norms=norms, y=y, f0=J)
        # update
        x += a * d
        # residual
        r = M * x - y
        rd = [D * x for D in Ds]
        # criterion
        J_old = copy(J)
        J = criterion(hypers=hypers, norms=norms, Ds=Ds, r=r, rd=rd)
        #resid = J / Jnorm
        #resid = (J_old - J) / Jnorm
        resid = np.abs(J_old - J) / Jnorm
        #resid = np.max(np.abs(d)) / np.max(np.abs(g0))
        callback(x)
    # define output
    if resid > tol:
        info = resid
    else:
        info = 0
    return x#, info

def criterion(x=None, y=None, M=None, norms=None, hypers=None, Ds=None,
              r=None, rd=None, C=None):
    if r is None:
        r = M * x - y
    if C is not None:
        if norms[0] != norm2:
            raise NotImplemented
        else:
            J = np.dot(r.T, C * r)
    else:
        J = norms[0](r)
    if rd is None:
        rd = [D * x for D in Ds]
    J += np.sum([h * norm(el) for norm, h, el in zip(norms[1:], hypers, rd)])
    return J

def gradient(x=None, y=None, M=None, dnorms=None, hypers=None, Ds=None,
             r=None, rd=None, C=None):
    if r is None:
        r = M * x - y
    if C is not None:
        if (dnorms[0] is not dnorm2):
            raise NotImplemented
        else:
            g = M.T * C * dnorms[0](r)
    else:
        g = M.T * dnorms[0](r)
    if rd is None:
        rd = [D * x for D in Ds]
    drs = [h * D.T * dnorm(el) for dnorm, h, D, el in zip(dnorms[1:], hypers, Ds, rd)]
    for dr in drs:
        g += dr
    ng = norm2(g)
    return g, ng

def acg(M, y, Ds=[], hypers=[], **kwargs):
    "Approximate Conjugate gradient"
    norms = (norm2, ) * (len(Ds) + 1)
    dnorms = (dnorm2, ) * (len(Ds) + 1)
    return gacg(M, y, Ds, hypers, norms, dnorms, **kwargs)

def hacg(M, y, Ds=[], hypers=[], deltas=None, **kwargs):
    "Huber Approximate Conjugate gradient"
    if deltas is None:
        return acg(M, Ds, hypers, y, **kwargs)
    norms = [hnorm(delta) for delta in deltas]
    dnorms = [dhnorm(delta) for delta in deltas]
    return gacg(M, y, Ds, hypers, norms, dnorms, **kwargs)

def npacg(M, y, Ds=[], hypers=[], ps=[], **kwargs):
    "Norm p Approximate Conjugate gradient"
    norms = [normp(p) for p in ps]
    dnorms = [dnormp(p) for p in ps]
    return gacg(M, y, Ds, hypers, norms, dnorms, **kwargs)

# norms
try:
    from scipy.linalg.fblas import dnrm2
except ImportError:
    pass

if 'dnrm2' in locals():
    def norm2(x):
        return dnrm2(x) ** 2
else:
    def norm2(x):
        return np.dot(x.ravel().T, x.ravel())

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

# line search methods

def quadratic_optimal_step(d, g, M, hypers, Ds, C=None):
    a = -.5 * np.dot(d.T, g)
    if C is None:
        a /= norm2(M * d) + np.sum([h * norm2(D * d) for h, D in zip(hypers, Ds)])
    else:
        Md = M * d
        a /= np.dot(Md, C * Md) + np.sum([h * norm2(D * d) for h, D in zip(hypers, Ds)])
    return a

def backtracking_line_search(d, g, M, hypers, Ds, x,
                             norms=None, y=None, f0=None, maxiter=10, tau=.5):
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
    """
    Callback function to display algorithm status and store values.
    """
    def __init__(self, verbose=False, criterion=False, solution=False, save=True):
        self.iter_ = []
        self.resid = []
        if criterion:
            self.criterion = []
        else:
            self.criterion = False
        if solution:
            self.solution = []
        else:
            self.solution = False
        self.verbose = verbose
        self.save = save
    def __call__(self, x):
        import inspect
        parent_locals = inspect.stack()[1][0].f_locals
        self.iter_.append(parent_locals['iter_'])
        self.resid.append(parent_locals['resid'])
        if self.criterion is not False:
            self.criterion.append(parent_locals['J'])
        if self.solution is not False:
            self.solution.append(parent_locals['x'])
        if self.verbose:
            self.print_status()
        if self.save:
            self.tofile()
    def print_status(self):
        # print header at first iteartion
        if len(self.iter_) == 1:
            header = 'Iteration \t Residual'
            if self.criterion is not False:
                header += '\t Criterion'
                print(header)
        # print status
        report = "\t%i \t %e" % (self.iter_[-1], self.resid[-1])
        if self.criterion is not False:
            report += '\t %e' % (self.criterion[-1])
        print(report)
    def tofile(self):
        var_dict = {"iter":self.iter_[-1],
                    "resid":self.resid[-1]}
        if self.criterion:
            var_dict["criterion"] = self.criterion[-1]
        if self.solution:
            var_dict["solution"] = self.solution[-1]
        np.savez("/tmp/lo_gacg_save.npz", **var_dict)

def normalize_hyper(hyper, y, x):
    """
    Normalize hyperparamaters so that they are independent of pb size
    """
    nx = float(x.size)
    ny = float(y.size)
    return np.asarray(hyper) * ny / nx

# functions with optional dependencies

try:
    import scipy.sparse.linalg as spl
except ImportError:
    pass

if 'spl' in locals():
    from sparse import *
    del spl

try:
    import scikits.optimization
except ImportError:
    pass

if 'scikits' in locals():
    if 'optimization' in scikits.__dict__:
        from optimization import *

try:
    import pywt
except ImportError:
    pass

if 'pywt' in locals():
    from iterative_thresholding import *

