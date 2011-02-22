"""
Use the optimization scikits to define a set of optimizer and models
using LinearOperators.
"""
import numpy as np
import lo
from scikits.optimization.optimizer import StandardOptimizer
from lo.iterative import *

class VerboseStandardOptimizer(StandardOptimizer):
    def record_history(self, **kwargs):
        iter_ = kwargs['iteration']
        J = kwargs['new_value']
        a = kwargs.get('alpha_step', 0)
        if iter_ == 0:
            print("Iteration \t Criterion \t Step")
        print("\t%i \t %e \t %e" % (iter_, J, a))

    recordHistory=record_history

class Model():
    def __init__(self, M, b, Ds=[], hypers=[], norms=[], dnorms=[], **kargs):
        M = lo.aslinearoperator(M)
        Ds = [lo.aslinearoperator(D) for D in Ds]
        self.M = M
        self.b = b
        self.Ds = Ds
        self.hypers = hypers
        self.norms = norms
        self.dnorms = dnorms
        # to store intermediate computations
        self.residual = None
        self.priors = None
        self.last_x = None
    def __call__(self, x, b=None):
        if b is None:
            b = self.b
        self.compute_residual(x)
        self.compute_priors(x)
        out = self.norms[0](self.residual)
        for Dx, h, norm in zip(self.priors, self.hypers, self.norms[1:]):
            out += h * norm(Dx)
        return out
    def gradient(self, x):
        self.compute_residual(x)
        self.compute_priors(x)
        out = self.M.T * self.dnorms[0](self.residual)
        for D, Dx, h, dnorm in zip(self.Ds, self.priors, self.hypers, self.dnorms[1:]):
            out += h * D.T * dnorm(Dx)
        return out
    def compute_residual(self, x):
        if self.residual is None or not np.all(x == self.last_x):
            self.residual = self.M * x - self.b
        self.last_x = x
    def compute_priors(self, x):
        if self.priors is None or not np.all(x == self.last_x):
            self.priors = [D * x for D in self.Ds]
        self.last_x = x

class QuadraticModel(Model):
    def __init__(self, M, b, Ds=[], hypers=[], **kargs):
        n = len(Ds) + 1
        norms = n * (norm2,)
        dnorms = n * (dnorm2,)
        Model.__init__(self, M, b, Ds=Ds, hypers=hypers, norms=norms,
                       dnorms=dnorms, **kargs)

class HuberModel(Model):
    def __init__(self, M, b, Ds=[], hypers=[], **kargs):
        n = len(Ds) + 1
        deltas = kargs.pop("deltas", None)
        if deltas is None:
            deltas = n * (None,)
        norms = list()
        dnorms = list()
        for d in deltas:
            if d is None:
                norms.append(norm2)
                dnorms.append(norm2)
            else:
                norms.append(hnorm(d))
                dnorms.append(dhnorm(d))
        Model.__init__(self, M, b, Ds=Ds, hypers=hypers, norms=norms,
                       dnorms=dnorms, **kargs)

class QuadraticOptimalStep():
    def __init__(self, M, b, Ds=[], hypers=[]):
        M = lo.aslinearoperator(M)
        Ds = [lo.aslinearoperator(D) for D in Ds]
        self.M = M
        self.b = b
        self.Ds = Ds
        self.hypers = hypers
    def __call__(self, origin, function, state, **kwargs):
        d = state['direction']
        g = state['gradient']
        a = -.5 * np.dot(d.T, g)
        a /= (norm2(self.M * d) + sum([h * norm2(D * d)
                                       for h, D in zip(self.hypers, self.Ds)]))
        state['alpha_step'] = a
        return origin + a * d

class BacktrackingFromOptimal():
    def __init__(self, M, b, Ds=[], hypers=[], rho=0.1, alpha_step=1.,
                 alpha_factor=0.5, **kwargs):
        from scikits.optimization import line_search

        self.optimal = QuadraticOptimalStep(M, b, Ds=Ds, hypers=hypers)
        self.backtracking = line_search.BacktrackingSearch(rho, alpha_step, alpha_factor)
    def __call__(self, origin, function, state, **kwargs):
        self.optimal(origin, function, state, **kwargs)
        state['initial_alpha_step'] = state['alpha_step']
        return self.backtracking(origin, function, state, **kwargs)

def quadratic_optimization(M, b, Ds=[], hypers=[], maxiter=None, tol=1e-6,
                           min_alpha_step=1e-10, **kwargs):
    """
    Use scikits.optimization to perform least-square inversion.
    """
    from scikits.optimization import step, line_search, criterion, optimizer

    if maxiter is None:
        maxiter = M.shape[0]
    model = QuadraticModel(M, b, Ds, hypers)
    mystep = step.PRPConjugateGradientStep()
    mylinesearch = QuadraticOptimalStep(M, b, Ds, hypers)
    #mylinesearch = line_search.QuadraticInterpolationSearch(
    #    min_alpha_step=min_alpha_step,alpha_step = 1e-4)
    mycriterion = criterion.criterion(gtol=tol, iterations_max=maxiter)
    myoptimizer = VerboseStandardOptimizer(function=model,
                                              step=mystep,
                                              line_search=mylinesearch,
                                              criterion=mycriterion,
                                              x0 = np.zeros(M.shape[1]))
    return myoptimizer.optimize()

def huber_optimization(M, b, Ds=[], hypers=[], maxiter=None, tol=1e-6,
                           min_alpha_step=1e-10, **kwargs):
    """
    Use scikits.optimization to perform least-square inversion.
    """
    from scikits.optimization import step, line_search, criterion, optimizer

    if maxiter is None:
        maxiter = M.shape[0]
    model = HuberModel(M, b, Ds, hypers, **kwargs)
    mystep = step.PRPConjugateGradientStep()
    mylinesearch = BacktrackingFromOptimal(M, b, Ds, hypers)
    mycriterion = criterion.criterion(ftol=tol, iterations_max=maxiter)
    myoptimizer = VerboseStandardOptimizer(function=model,
                                              step=mystep,
                                              line_search=mylinesearch,
                                              criterion=mycriterion,
                                              x0 = np.zeros(M.shape[1]))
    return myoptimizer.optimize()
