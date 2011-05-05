"""
Implement algorithms using the LinearOperator class
"""
import numpy as np
from copy import copy

from norms import *
from linesearch import *
from criterions import *
from algorithms import *

# functions with optional dependencies


try:
    import scikits.optimization
except ImportError:
    pass

if 'scikits' in locals():
    if 'optimization' in scikits.__dict__:
        from optimization import *

try:
    from scipy import optimize as scipy_optimize
except ImportError:
    pass

if "scipy_optimize" in locals():
    from optimize import *
