"""
Implement algorithms using the LinearOperator class
"""
import numpy as np
from copy import copy
import lo

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

class Norm(object):
    def __call__(self, x):
        return self._call(x)
    def diff(self, x):
        return self._diff(x)

class Norm2(Norm):
    def __init__(self, C=None):
        if C is None:
            def call(x):
                return norm2(x)
            def diff(x):
                return 2 * x
        else:
            def call(x):
                return np.dot(r.T, C * r)
            def diff(x):
                return 2 * C * r
        self.C = C
        self._call = call
        self._diff = diff

class Huber(Norm):
    def __init__(self, delta):
        self.delta = delta
        self._call = hnorm(d=delta)
        self._diff = dhnorm(d=delta)

class Normp(Norm):
    def __init__(self, p):
        self.p = p
        self._call = normp(p=p)
        self._diff = dnormp(p=p)

# criterions

class Criterion(object):
    """
    A class representing criterions such as :
    ..math: || y - H x ||^2 + sum_i \hypers_i || D_i x ||^2

    Parameters
    ----------
    H : LinearOperator
        Model.
    y : ndarray
        Data array.
    hypers: tuple or array
        Hypeparameter of each prior.
    Ds : tuple of LinearOperators
        Prior models.
    norms : tuple
        Can be norm2, huber, normp
    store : boolean (default: True)
        Store last criterion computation.
    """
    def __init__(self, model, data, hypers=[], priors=[], norms=[], store=True):
        self.model = model
        self.data = data
        self.priors = priors
        # normalize hyperparameters
        self.hypers = np.asarray(hypers) * model.shape[0] / float(model.shape[1])
        # default to Norm2
        self.norms = norms
        if len(self.norms) == 0:
            self.norms = (Norm2(), ) * (len(self.priors) + 1)
        # get diff of norms
        self.dnorms = [n.diff for n in self.norms]
        # to store intermediate values
        self.store = store
        self.last_x = None
        self.projection = None
        self.last_residual = None
        self.last_prior_residuals = None
        # if all norms are l2 define optimal step
        self._optimal_step = np.all([n == Norm2 for n in self.norms])
        # set number of unknowns
        self.n_variables = self.model.shape[1]
    def islast(self, x):
        return np.all(x == self.last_x)
    def load_last(self):
        return self.projection, self.last_residual, self.last_prior_residuals
    def get_residuals(self, x):
        if self.islast(x):
            Hx, r, rd = self.load_last()
        else:
            Hx = self.model * x
            r = Hx - self.data
            rd = [D * x for D in self.priors]
        return Hx, r, rd
    def save(self, x, Hx, r, rd):
        if self.store and not self.islast(x):
            self.last_x = copy(x)
            self.Hx = copy(Hx)
            self.last_residual = copy(r)
            self.last_prior_residuals = copy(rd)
    def __call__(self, x):
        # residuals
        Hx, r, rd = self.get_residuals(x)
        # norms
        J = self.norms[0](r)
        priors = [norm(rd_i) for norm, rd_i in zip(self.norms[1:], rd)]
        J += np.sum([h * prior for h, prior in zip(self.hypers, priors)])
        self.save(x, Hx, r, rd)
        return J
    def gradient(self, x):
        """
        First order derivative of the criterion as a function of x.
        """
        Hx, r, rd = self.get_residuals(x)
        g = self.model.T * self.dnorms[0](r)
        p_dnorms = [dnorm(el) for dnorm, el in zip(self.dnorms[1:], rd)]
        p_diff = [D.T * dn for D, dn in zip(self.priors, p_dnorms)]
        drs = [h * pd for h, pd in zip(self.hypers, p_diff)]
        for dr in drs:
            g += dr
        self.save(x, Hx, r, rd)
        return g

class QuadraticCriterion(Criterion):
    def __init__(self, model, data, hypers=[], priors=[], store=True):
        norms = (Norm2(), ) * (len(priors) + 1)
        Criterion.__init__(self, model, data, hypers=hypers, priors=priors, store=store)

class HuberCriterion(Criterion):
    def __init__(self, model, data, hypers=[], deltas=[], priors=[], store=True):
        norms = [Huber(d) for d in deltas]
        Criterion.__init__(self, model, data, hypers=hypers, priors=priors, store=store)

# update types

def fletcher_reeves(algo):
    return algo.current_gradient_norm / algo.last_gradient_norm

def polak_ribiere(algo):
    b =  np.dot(algo.current_gradient.T,
                (algo.current_gradient - algo.last_gradient))
    b /= np.norm(algo.last_gradient)
    return b

# line searches

def optimal_step(algo):
    # get variables from criterion
    d = algo.current_descent
    g = algo.current_gradient
    H = algo.criterion.model
    Ds = algo.criterion.priors
    hypers = algo.criterion.hypers
    norms = algo.criterion.norms
    # replace norms by Norm2 if not a Norm2 instance
    # to handle properly Norm2 with C covariance matrices ...
    norms = [n if isinstance(n, Norm2) else Norm2() for n in norms]
    # compute quadratic optimal step
    a = -.5 * np.dot(d.T, g)
    a /= norm2(H * d) + np.sum([h * norm2(D * d) for h, D in zip(hypers, Ds)])
    return a

class Backtracking(object):
    def __init__(self, maxiter=10, tau=.5):
        self.maxiter = maxiter
        self.tau = tau
    def __call__(self, algo):
        x = algo.current_solution
        d = algo.current_descent
        a = optimal_step(algo)
        i = 0
        f0 = algo.current_criterion
        fi = 2 * f0
        while (i < self.maxiter) and (fi > f0):
            i += 1
            a *= tau
            xi = x + a * d
            fi = algo.criterion(xi)
        return a

default_backtracking = Backtracking()

# algorithms

class ConjugateGradient(object):
    """
    Apply the conjugate gradient algorithm to a Criterion instance
    """
    def __init__(self, criterion, x0=None, tol=1e-6, maxiter=None,
                 verbose=False, savefile=None, update_type=fletcher_reeves,
                 line_search=optimal_step, **kwargs):
        self.criterion = criterion
        self.gradient = criterion.gradient
        self.n_variables = self.criterion.n_variables
        self.tol = tol
        if maxiter is None:
            self.maxiter = self.n_variables
        else:
            self.maxiter = maxiter
        self.savefile = savefile
        self.verbose = verbose
        self.update_type = update_type
        self.line_search = line_search
        self.kwargs = kwargs
        # to store values
        self.current_criterion = None
        self.current_solution = None
        self.current_gradient = None
        self.current_gradient_norm = None
        self.current_descent = None
        self.last_criterion = None
        self.last_solution = None
        self.last_gradient = None
        self.last_gradient_norm = None
        self.last_descent = None
        self.optimal_step = None
    def initialize(self):
        self.first_guess()
        self.first_criterion = self.criterion(self.current_solution)
        self.current_criterion = self.first_criterion
        self.resid = 2 * self.tol
        self.iter_ = 0
    def first_guess(self, x0=None):
        if x0 is None:
            self.current_solution = np.zeros(self.n_variables)
        else:
            self.current_solution = copy(x0)
    def stop_condition(self):
        return self.iter_ < self.maxiter and self.resid > self.tol
    def update_gradient(self):
        self.last_gradient = copy(self.current_gradient)
        self.current_gradient = self.gradient(self.current_solution)
    def update_gradient_norm(self):
        self.last_gradient_norm = copy(self.current_gradient_norm)
        self.current_gradient_norm = norm2(self.current_gradient)
    def update_descent(self):
        if self.iter_ == 1:
            self.current_descent = - self.current_gradient
        else:
            self.last_descent = copy(self.current_descent)
            b = self.update_type(self)
            self.current_descent = - self.current_gradient + b * self.last_descent
    def update_solution(self):
        self.last_solution = copy(self.current_solution)
        a = self.line_search(self)
        self.current_solution += a * self.current_descent
    def update_criterion(self):
        self.last_criterion = copy(self.current_criterion)
        self.current_criterion = self.criterion(self.current_solution)
    def update_resid(self):
        self.resid = np.abs(self.last_criterion - self.current_criterion)
        self.resid /= self.first_criterion
    def update(self):
        self.update_gradient()
        self.update_gradient_norm()
        self.update_descent()
        self.update_solution()
        self.update_criterion()
        self.update_resid()
    def print_status(self):
        if self.verbose:
            if self.iter_ == 1:
                print('Iteration \t Residual \t Criterion')
            print("\t%i \t %e \t %e" %
                  (self.iter_, self.resid, self.current_criterion))
    def save(self):
        if self.savefile is not None:
            var_dict = {
                "iter":self.iter_,
                "resid":self.resid,
                "criterion":self.current_criterion,
                "solution":self.current_solution,
                "gradient":self.current_gradient
                }
            np.savez(self.savefile, **var_dict)
    def iterate(self):
        self.iter_ += 1
        self.update()
        self.print_status()
        self.save()
        # return value not used in loop but usefull in "interactive mode"
        return self.current_solution
    def __call__(self):
        self.initialize()
        while self.stop_condition():
            self.iterate()
        return self.current_solution

class QuadraticConjugateGradient(ConjugateGradient):
    def __init__(self, model, data, priors=[], hypers=[], **kwargs):
        store = kwargs.pop("store", True)
        criterion = QuadraticCriterion(model, data, hypers=hypers,
                                       priors=priors, store=store)
        ConjugateGradient.__init__(self, criterion, **kwargs)

class HuberConjugateGradient(ConjugateGradient):
    def __init__(self, model, data, priors=[], hypers=[], deltas=None, **kwargs):
        store = kwargs.pop("store", True)
        criterion = HuberCriterion(model, data, hypers=hypers, priors=priors,
                                   deltas=deltas, store=store)
        ConjugateGradient.__init__(self, criterion, **kwargs)
 
# for backward compatibility

def acg(model, data, priors=[], hypers=[], **kwargs):
    algorithm = QuadraticConjugateGradient(model, data, priors=priors,
                                           hypers=hypers, **kwargs)
    sol = algorithm()
    return sol

def hacg(model, data, priors=[], hypers=[], deltas=None, **kwargs):
    algorithm = HuberConjugateGradient(model, data, priors=priors,
                                       hypers=hypers, deltas=deltas, **kwargs)
    sol = algorithm()
    return sol

# other

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

