"""
Use the optimization scikits to define a set of optimizer and models
using LinearOperators.
"""
import numpy as np
import lo
from scikits.optimization import step, line_search, criterion, optimizer
from scikits.optimization.optimizer import StandardOptimizer
import criterions


# defaults
default_step = step.PRPConjugateGradientStep()
default_line_search = line_search.QuadraticInterpolationSearch(0)
default_stop_criterion = criterion.criterion()

# a class wrapping StandardOptimizer
class Optimizer(object):
    def __init__(self, criterion,
                 x0=None,
                 step=default_step,
                 line_search=default_line_search,
                 stop_criterion=default_stop_criterion):
        self.criterion = criterion
        self.n_variables = criterion.n_variables
        self.step = step
        self.line_search = line_search
        self.stop_criterion = stop_criterion
        self.first_guess()
        self.optimizer = StandardOptimizer(function=self.criterion,
                                           step=self.step,
                                           line_search=self.line_search,
                                           criterion=self.stop_criterion,
                                           x0=self.current_solution)

    def first_guess(self, x0=None):
        """
        Sets current_solution attribute to initial value.
        """
        if x0 is None:
            self.current_solution = np.zeros(self.n_variables)
        else:
            self.current_solution = copy(x0)
    def iterate(self):
        self.optimizer.iterate()
    def __call__(self):
        self.optimizer.optimize()

class QuadraticOptimizer(Optimizer):
    def __init__(self, model, data, priors, hypers, **kwargs):
        criterion = criterions.QuadraticCriterion(model, data, hypers=hypers,
                                                  priors=priors)
        Optimizer.__init__(self, criterion, **kwargs)
