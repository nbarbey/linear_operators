"""
Implements iterative algorithm class.
"""
import numpy as np
from copy import copy
import lo

from linesearch import *
from criterions import *

# update types

def fletcher_reeves(algo):
    """
    Fletcher-Reeves descent direction update method.
    """
    return algo.current_gradient_norm / algo.last_gradient_norm

def polak_ribiere(algo):
    """
    Polak-Ribiere descent direction update method.
    """
    b =  np.dot(algo.current_gradient.T,
                (algo.current_gradient - algo.last_gradient))
    b /= np.norm(algo.last_gradient)
    return b

# algorithms

class ConjugateGradient(object):
    """
    Apply the conjugate gradient algorithm to a Criterion instance.

    Parameters
    ----------

    criterion : Criterion
        A Criterion instance. It should have following methods and attributes:
            __call__ : returns criterion values at given point
            gradient : returns gradient (1st derivative) of criterion at given point
            n_variable: the size of the input vector of criterion

    x0 : ndarray (None)
        The first guess of the algorithm.

    tol : float (1e-6)
        The tolerance. The algorithm will stop if the residual is below the tolerance.

    maxiter : int (None)
        Maximal number of iterations.

    verbose : boolean (False)
        Print information at each iteration.

    savefile : string (None)
        Save current result to a file (npy / npz format)

    update_type : function (fletcher_reeves)
        Type of descent direction update : e.g. fletcher_reeves, polak_ribiere

    line_search : function (optimal step)
        Line search method to find the minimum along each direction at each
        iteration.

    Returns
    -------

    Returns an algorithm instance. Optimization is performed by
    calling the this instance.

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
        """
        Initialize required values.
        """
        self.first_guess()
        self.first_criterion = self.criterion(self.current_solution)
        self.current_criterion = self.first_criterion
        self.resid = 2 * self.tol
        self.iter_ = 0
    def first_guess(self, x0=None):
        """
        Sets current_solution attribute to initial value.
        """
        if x0 is None:
            self.current_solution = np.zeros(self.n_variables)
        else:
            self.current_solution = copy(x0)
    def stop_condition(self):
        """
        Returns False when algorithm should stop.
        """
        return self.iter_ < self.maxiter and self.resid > self.tol
    # update_* functions encode the actual algorithm
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
        """
        Update all values.
        """
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
        """
        Perform one iteration and returns current solution.
        """
        self.iter_ += 1
        self.update()
        self.print_status()
        self.save()
        # return value not used in loop but usefull in "interactive mode"
        return self.current_solution
    def __call__(self):
        """
        Perform the optimization.
        """
        self.initialize()
        while self.stop_condition():
            self.iterate()
        return self.current_solution

class QuadraticConjugateGradient(ConjugateGradient):
    """
    A subclass of ConjugateGradient using a QuadraticCriterion.
    """
    def __init__(self, model, data, priors=[], hypers=[], **kwargs):
        store = kwargs.pop("store", True)
        criterion = QuadraticCriterion(model, data, hypers=hypers,
                                       priors=priors, store=store)
        ConjugateGradient.__init__(self, criterion, **kwargs)

class HuberConjugateGradient(ConjugateGradient):
    """
    A subclass of ConjugateGradient using an HuberCriterion.
    """
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
