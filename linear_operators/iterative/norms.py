"""
Implement norm classes. Available now :

- Norm2
- Huber norm
- Norm-k
"""
import numpy as np

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
    """
    An abstract class to define norm classes.
    """
    def __call__(self, x):
        return self._call(x)
    def diff(self, x):
        return self._diff(x)

class Norm2(Norm):
    """
    A norm-2 class. Optionally accepts a covariance matrix C.
    If C is given, the norm would be : np.dot(x.T, C * x).
    Otherwise, it would be norm2(x).

    Parameters
    ----------

    C : LinearOperator (None)
        The covariance matrix of the norm.

    Returns
    -------
    Returns a Norm2 instance with a __call__ and a diff method.
    """
    def __init__(self, C=None):
        if C is None:
            def call(x):
                return norm2(x)
            def diff(x):
                return 2 * x
        else:
            def call(x):
                return np.dot(x.T, C * x)
            def diff(x):
                return 2 * C * x
        self.C = C
        self._call = call
        self._diff = diff

class Huber(Norm):
    """
    An Huber norm class.

    Parameters
    ----------

    delta: float
       The Huber parameter of the norm.
       if abs(x_i) is below delta, returns x_i ** 2
       else returns 2 * delta * x_i - delta ** 2

    Returns
    -------
    Returns an Huber instance with a __call__ and a diff method.
     """
    def __init__(self, delta):
        self.delta = delta
        self._call = hnorm(d=delta)
        self._diff = dhnorm(d=delta)

class Normp(Norm):
    """
    An Norm-p class.

    Parameters
    ----------

    p: float
       The power of the norm.
       The norm will be np.sum(np.abs(x) ** p)

    Returns
    -------
    Returns a Normp instance with a __call__ and a diff method.
     """
    def __init__(self, p):
        self.p = p
        self._call = normp(p=p)
        self._diff = dnormp(p=p)
