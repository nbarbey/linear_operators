"""
Implements Double loop inference algorithms.

Reference
---------

Bayesian Inference and Optimal Design for the Sparse Linear Model,
Matthias W. Seeger

http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.165.8284&rep=rep1&type=pdf

"""
from copy import copy
import numpy as np
import numexpr as ne
from utils import eigendecomposition
from algorithms import Algorithm, default_callback, StopCondition
from ..operators import diag
from optimize import *
from norms import norm2, dnorm2

default_stop = StopCondition(maxiter=5)

class Criterion(object):
    def __init__(self, algo):
        self.algo = algo
        self.n_variables = self.algo.model.shape[1]
        # storing
        self.last_u = None
        self.Xu = None
        self.Bu = None
    def islast(self, u):
        return np.all(u == self.last_u)
    def load_last(self):
        return self.Xu, self.Bu
    def get_projections(self, u):
        if self.islast(u):
            return self.load_last()
        else:
            self.last_u = copy(u)
            X = self.algo.model 
            B = self.algo.prior
            self.Xu = X * u
            self.Bu = B * u
            return self.Xu, self.Bu
    def likelihood(self, u):
        sigma = self.algo.sigma
        X = self.algo.model
        y = self.algo.data
        Xu, Bu = self.get_projections(u)
        return sigma ** (-2) * norm2(Xu - y)
    def dlike(self, u):
        sigma = self.algo.sigma
        X = self.algo.model
        y = self.algo.data
        Xu, Bu = self.get_projections(u)
        return sigma ** (-2) * X.T * dnorm2(Xu - y)
    def d2like(self, u):
        sigma = self.algo.sigma
        X = self.algo.model
        return sigma ** (-2) * X.T * X
    def d2lik_p(self, u, p):
        return self.d2like(u) * p
    def penalization(self, u):
        sigma = self.algo.sigma
        B = self.algo.prior
        t = self.algo.tau
        z = self.algo.z
        #e = t * np.sqrt(z + (np.abs(B * u) / sigma) ** 2)
        Xu, Bu = self.get_projections(u)
        e = ne.evaluate("2 * t * sqrt(z + (abs(Bu) / sigma) ** 2)")
        return e.sum()
    def dpen(self, u):
        sigma = self.algo.sigma
        B = self.algo.prior
        t = self.algo.tau
        z = self.algo.z
        Xu, Bu = self.get_projections(u)
        #e = 2 * (t * Bu) / np.sqrt(z + (Bu / sigma) ** 2)
        e = ne.evaluate("2 * (t * Bu) / sqrt(z + (Bu / sigma) ** 2)")
        return (B.T * e) / (sigma ** 2)
    def d2pen(self, u):
        sigma = self.algo.sigma
        B = self.algo.prior
        t = self.algo.tau
        z = self.algo.z
        Xu, Bu = self.get_projections(u)
        rho = ne.evaluate("(t * z) / ((z + (Bu / sigma) ** 2) ** (1.5) * sigma ** 2)")
        return B.T * diag(rho) * B
    def d2pen_p(self, u, p):
        return self.d2pen(u) * p
    def __call__(self, u):
        return self.likelihood(u) + self.penalization(u)
    def gradient(self, u):
        return self.dlike(u) + self.dpen(u)
    def hessian(self, u):
        return self.d2like(u) + self.d2pen(u)
    def hessian_p(self, u, p):
        return self.hessian(u) * p

class DoubleLoopAlgorithm(Algorithm):
    """
    A class implementing the double loop algorithm.

    Parameters
    ----------

    model : LinearOperator
        Linear model linking data and unknowns.
    data : ndarray
        Data.
    prior : LinearOperator
        Prior.
    tau : ndarray
        Parameters of the Laplace potential on priors coefficients.
    sigma : float
        Likelihood standard deviation.
    lanczos : dict
        Keyword arguments of the Lanczos decomposition.
    fmin_args : dict
        Keyword arguments of the function minimization.
    """
    def __init__(self, model, data, prior, tau, sigma, optimizer=FminSLSQP,
                 lanczos={}, fmin_args={},
                 callback=default_callback,
                 stop_condition=default_stop,
                 ):
        self.model = model
        self.data = data
        self.prior = prior
        self.tau = tau
        self.sigma = sigma
        self.optimizer = optimizer
        self.lanczos = lanczos
        self.fmin_args = fmin_args
        # 
        self.callback = callback
        self.stop_condition = stop_condition
        # to store internal variables
        self.z = None
        self.gamma = None
        self.inv_gamma = None
        self.g_star = None
        self.current_solution = None
        self.last_solution = None
        self.inv_cov = None
        self.inv_cov_approx = None
        self.criterion = None
    def initialize(self):
        """
        Set parameters to initial values.
        """
        self.z = 0.05 * np.ones(self.model.shape[1])
        self.g_star = 0.
        self.current_solution = np.zeros(self.model.shape[1])
        self.iter_ = 0
        self.gamma = np.ones(self.prior.shape[0])
        self.update_inv_gamma()
    def iterate(self):
        print "outer"
        self.outer()
        print "inner"
        self.inner()
        return Algorithm.iterate(self)
    # outer loop
    def outer(self):
        """
        Outer loop : Lanczos approximation.
        """
        self.update_inv_cov()
        self.update_inv_cov_approx()
        self.update_z()
        self.update_g_star()
    def update_inv_cov(self):
        D = diag(self.gamma ** (-1), dtype=self.prior.dtypeout)
        X = self.model
        B = self.prior
        self.inv_cov = X.T * X + B.T * D * B
    def update_inv_cov_approx(self):
        self.inv_cov_approx = eigendecomposition(self.inv_cov, **self.lanczos)
    def update_z(self):
        # get eigenvalues, eigenvectors
        e = self.inv_cov_approx.eigenvalues
        v = self.inv_cov_approx.eigenvectors
        B = self.prior
        self.z = sum([ei * (B * vi) ** 2 for ei, vi in zip(e, v.T)])
    def update_g_star(self):
        self.g_star = np.dot(self.z.T, self.inv_gamma)
        self.g_star -= self.inv_cov_approx.logdet()
    # inner loop
    def inner(self):
        """
        Inner loop : Penalized minimization.
        """
        self.update_current_solution()
        self.update_gamma()
        self.update_inv_gamma()
    def update_current_solution(self):
        self.inner_criterion = Criterion(self)
        self.last_solution = copy(self.current_solution)
        self.inner_algo = self.optimizer(self.inner_criterion,
                                         self.last_solution,
                                         **self.fmin_args)
        self.current_solution = self.inner_algo()
    def update_gamma(self):
        s = np.abs(self.prior * self.current_solution)
        sn2 = (s / self.sigma) ** 2
        self.gamma = np.sqrt(self.z + sn2) / self.tau
    def update_inv_gamma(self):
        self.inv_gamma = self.gamma ** (-1)
