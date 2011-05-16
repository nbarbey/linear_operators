"""
Implement the criterion class. Available :

- Criterion
- QuadraticCriterion
- HuberCriterion
"""
from copy import copy
import numpy as np
from norms import *

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

    Returns
    -------

    Returns an Criterion instance with __call__ and gradient methods.
    """
    def __init__(self, model, data, hypers=[], priors=[], norms=[], store=True):
        self.model = model
        self.data = data.ravel()
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
    """
    Subclass of Criterion with all norms forced to be Norm2 instances.
    """
    def __init__(self, model, data, hypers=[], priors=[], store=True):
        norms = (Norm2(), ) * (len(priors) + 1)
        Criterion.__init__(self, model, data, hypers=hypers, priors=priors, store=store)

class HuberCriterion(Criterion):
    """
    Subclass of Criterion with all norms forced to be Huber instances.
    """
    def __init__(self, model, data, hypers=[], deltas=[], priors=[], store=True):
        norms = [Huber(d) for d in deltas]
        Criterion.__init__(self, model, data, hypers=hypers, priors=priors, store=store)
