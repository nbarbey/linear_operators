"""
Implement algorithms using the LinearOperator class
"""
import numpy as np
from copy import copy
import lo

from norms import *
from linesearch import *
from criterions import *
from iterative_algorithms import *

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

try:
    from scipy import optimize as scipy_optimize
except ImportError:
    pass

if "scipy_optimize" in locals():
    from optimize import *
