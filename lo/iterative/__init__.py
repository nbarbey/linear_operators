"""
Implement algorithms using the LinearOperator class
"""
try:
    import scipy.sparse.linalg as spl
except ImportError:
    pass

if 'spl' in locals():
    from iterative import *
    del spl

try:
    import scikits.optimization
except ImportError:
    pass

if 'scikits.optimization' in locals():
    from optimization import *

try:
    import pywt
except ImportError:
    pass

if 'pywt' in locals():
    from iterative_thresholding import *
