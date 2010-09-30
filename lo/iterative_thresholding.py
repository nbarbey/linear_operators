"""
Use the pywt package to perform iterative thresholding algorithms.
"""
import lo
import pywt
from pywt import thresholding

# Iterative Thresholding methods

def landweber(A, W, y, mu=1., nu=None, threshold=thresholding.hard, x0=None,
              maxiter=100, callback=None):
    """Landweber algorithm
    
    Input
    -----
    A : measurement matrix
    W : wavelet transform
    y : data
    mu : step of the gradient update
    nu : thresholding coefficient
    threshold : thresholding function
    maxiter : number of iterations
    callback : callback function
    
    Output
    ------
    
    x : solution
    
    """
    if callback is None:
        callback = lo.CallbackFactory(verbose=True)
    if x0 is None:
        x = A.T * y
    else:
        x = copy(x0)
    for iter_ in xrange(maxiter):
        r = A * x - y
        x += .5 * mu * A.T * (r)
        x = W.T * threshold(W * x, nu)
        resid = lo.norm2(r) + 1 / (2 * nu) * lo.normp(p=1)(x)
        callback(x)
    return x

def fista(A, W, y, mu=1., nu=None, threshold=thresholding.hard, x0=None,
          maxiter=100, callback=None):
    """ Fista algorithm
    
    """
    if callback is None:
        callback = lo.CallbackFactory(verbose=True)
    if x0 is None:
        x = A.T * y
    else:
        x = copy(x0)
    x_old = np.zeros(x.shape)
    t_old = 1.
    for iter_ in xrange(maxiter):
        t = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
        a = (t_old - 1) / t
        t_old = copy(t)
        z = x + a * (x - x_old)
        g = A.T * (A * z - y)
        x = z - .5 * mu * g
        x = W.T * threshold(W * x, nu)
        x_old = copy(x)
        resid = lo.norm2(A * x - y) + 1 / (2 * nu) * lo.normp(p=1)(x)
        callback(x)
    return x

