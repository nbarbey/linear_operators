"""
Use the optimization scikits to define a set of optimizer and models
using LinearOperators.
"""
import numpy as np
import lo
from scikits.optimization.optimizer import StandardOptimizer
from lo.iterative import norm2, dnorm2

class QuadraticModel():
    def __init__(self, M, b, Ds=[], hypers=[]):
        M = lo.aslinearoperator(M)
        Ds = [lo.aslinearoperator(D) for D in Ds]
        self.M = M
        self.b = b
        self.Ds = Ds
        self.hypers = hypers
    def __call__(self, x, b=None):
        if b is None:
            b = self.b
        out = norm2(self.M * x - b)
        out += np.sum([h * norm2(D * x) 
                       for D, h in zip(self.Ds, self.hypers)])
        return out
    def gradient(self, x):
        r = self.M * x - self.b
        out = self.M.T * dnorm2(r)
        drs = [h * D.T * dnorm2(D * x) for D, h in zip(self.Ds, self.hypers)]
        for dr in drs:
            out += dr
        return out

class VerboseStandardOptimizer(StandardOptimizer):
    def record_history(self, **kwargs):
        print "iteration : " + str(kwargs['iteration'])
        print "current value: " + str(kwargs['new_value'])
        print "step: " + str(kwargs.get('alpha_step'))

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
        a /= norm2(self.M * d) + np.sum([h * norm2(D * d)
                                         for h, D in zip(self.hypers, self.Ds)])
        state['alpha_step'] = a
        return origin + a * d

def opt(M, b, Ds=[], hypers=[], maxiter=None, tol=1e-6, min_alpha_step=1e-10):
    """
    Use scikits.optimization to perform least-square inversion.
    """
    from scikits.optimization import step, line_search, criterion, optimizer

    if maxiter is None:
        maxiter = M.shape[0]
    model = QuadraticModel(M, b, Ds, hypers)
    mystep = step.PRPConjugateGradientStep()
    mylinesearch = QuadraticOptimalStep(M, b, Ds, hypers)
    mycriterion = criterion.criterion(ftol=tol, iterations_max=maxiter)
    myoptimizer = VerboseStandardOptimizer(function=model, 
                                              step=mystep,
                                              line_search=mylinesearch,
                                              criterion=mycriterion,
                                              x0 = np.zeros(M.shape[1]))
    return myoptimizer.optimize()
