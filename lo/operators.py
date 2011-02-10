"""Definition of useful linear operators"""
import numpy as np
from copy import copy
from interface import LinearOperator

class NDOperator(LinearOperator):
    """Subclass of LinearOperator that handle multidimensional inputs and outputs"""
    def __init__(self, shapein, shapeout, matvec, rmatvec=None, matmat=None, rmatmat=None,
                 dtypein=None, dtypeout=None, dtype=None):

        sizein = np.prod(shapein)
        sizeout = np.prod(shapeout)
        shape = (sizeout, sizein)

        ndmatvec = lambda x: matvec(x.reshape(shapein)).ravel()

        if rmatvec is not None:
            ndrmatvec = lambda x: rmatvec(x.reshape(shapeout)).ravel()
        else:
            ndrmatvec = None

        LinearOperator.__init__(self, shape, ndmatvec, ndrmatvec, dtype=dtype,
                                dtypein=dtypein, dtypeout=dtypeout)

        self.ndmatvec = matvec
        self.ndrmatvec = rmatvec
        self.shapein = shapein
        self.shapeout = shapeout


class NDSOperator(NDOperator):
    def __init__(self, shapein=None, shapeout=None, classin=None,
                 classout=None, dictin=None, dictout=None, xin=None, xout=None,
                 matvec=None, rmatvec=None, dtype=np.float64, dtypein=None,
                 dtypeout=None):
        "Wrap linear operation working on ndarray subclasses in InfoArray style"
        if xin is not None:
            shapein = xin.shape
            classin = xin.__class__
            dictin = xin.__dict__
            dtype = xin.dtype

        if xout is not None:
            shapeout = xout.shape
            classout = xout.__class__
            dictout = xout.__dict__

            sizein = np.prod(shapein)
            sizeout = np.prod(shapeout)
            shape = (sizeout, sizein)

        self.ndsmatvec = matvec
        self.ndsrmatvec = rmatvec
        self.classin = classin
        self.classout = classout
        self.dictin = dictin
        self.dictout = dictout
        self.shapein = shapein
        self.shapeout = shapeout

        if matvec is not None:
            def smatvec(x):
                xi = classin(data=x)
                xi.__dict__ = dictin
                return matvec(xi)
        else:
            raise ValueError('Requires a matvec function')

        if rmatvec is not None:
            def srmatvec(x):
                xo = classout(data=x)
                xo.__dict__ = dictout
                return rmatvec(xo)
        else:
            rmatvec = None

        NDOperator.__init__(self, shapein, shapeout, smatvec, rmatvec=srmatvec,
                            dtypein=dtypein, dtypeout=dtypeout, dtype=dtype)
        
def ndoperator(*kargs, **kwargs):
    "Transform n-dimensional linear operators into LinearOperators"
    return NDOperator(*kargs, **kwargs)

def masubclass(xin=None, xout=None, shapein=None, shapeout=None, classin=None,
               classout=None, dictin=None, dictout=None,
               matvec=None, rmatvec=None, dtype=np.float64, dtypein=None, dtypeout=None):
    "Wrap linear operation working on ndarray subclasses in MaskedArray style"
    if xin is not None:
        shapein = xin.shape
        classin = xin.__class__
        dictin = xin.__dict__
        dtype = xin.dtype
    if xout is not None:
        shapeout = xout.shape
        classout = xout.__class__
        dictout = xout.__dict__
    sizein = np.prod(shapein)
    sizeout = np.prod(shapeout)
    shape = (sizeout, sizein)
    if matvec is not None:
        def ndmatvec(x):
            xi = classin(x.reshape(shapein))
            xi.__dict__ = dictin
            return matvec(xi).reshape(sizeout)
    else:
        raise ValueError('Requires a matvec function')
    if rmatvec is not None:
        def ndrmatvec(x):
            xo = classout(x.reshape(shapeout))
            xo.__dict__ = dictout
            return rmatvec(xo).reshape(sizein)
    else:
        ndrmatvec = None
    return LinearOperator(shape, matvec=ndmatvec, rmatvec=ndrmatvec, dtype=dtype,
                          dtypein=dtypein, dtypeout=dtypeout)

def ndsubclass(**kwargs):
    "Wrap linear operation working on ndarray subclasses in InfoArray style"
    return NDSOperator(**kwargs)

def diag(d, shape=None, dtype=None):
    "Returns a diagonal Linear Operator"
    if shape is None:
        shape = 2 * (d.size,)
    if shape[0] != shape[1]:
        raise ValueError('Diagonal operators must be square')
    def matvec(x):
        return d * x
    if dtype is None:
        dtype = d.dtype
    return LinearOperator(shape, matvec=matvec, rmatvec=matvec, dtype=dtype)

def identity(shape, dtype=np.float64):
    "Returns the identity linear Operator"
    if shape[0] != shape[1]:
        raise ValueError('Identity operators must be square')
    def matvec(x):
        return x
    return LinearOperator(shape, matvec=matvec, rmatvec=matvec, dtype=dtype)

def eye(shape, dtype=np.float64):
    "Returns the identity linear Operator"
    if shape[0] == shape[1]:
       return identity(shape, dtype=dtype)
    else:
        def matvec(x):
            return x[:shape[0]]
        def rmatvec(x):
            return np.concatenate(x, np.zeros(shape[0] - shape[1]))
        return LinearOperator(shape, matvec=matvec, rmatvec=rmatvec, dtype=dtype)

def fftn(shapein, dtypein=np.float64, dtypeout=np.complex128, s=None, axes=None):
    "fftn LinearOperator"
    import numpy.fft
    if s is None:
        shapeout = shapein
    else:
        shapeout = s
    def matvec(x):
        return np.fft.fftn(x, s=s, axes=axes)
    def rmatvec(x):
        return np.fft.ifftn(x, s=s, axes=axes)
    return ndoperator(shapein, shapeout, matvec, rmatvec, dtypein=dtypein,
                      dtypeout=dtypeout)

def fft2(shapein, dtypein=np.float64, dtypeout=np.complex128, s=None, axes=(-2, -1)):
    "fft2 LinearOperator"
    import numpy.fft
    if len(shapein) != 2:
        raise ValueError('Error expected 2 dimensional shape')
    if s is None:
        shapeout = shapein
    else:
        shapeout = s
    def matvec(x):
        return np.fft.fftn(x, s=s, axes=axes)
    def rmatvec(x):
        return np.fft.ifftn(x, s=s, axes=axes)
    return ndoperator(shapein, shapeout, matvec, rmatvec, 
                      dtypein=dtypein, dtypeout=dtypeout)

def convolve(shapein, kernel, mode='full'):
    """ Linear Operator to convolve two N-dimensional arrays

    See Also:
      scipy.signal.convolve
    """
    from scipy.signal import convolve
    #if len(shapein) != 2:
    #    raise ValueError('Error expected 2 dimensional shape')
    if mode == 'full':
        shapeout = [s + ks - 1 for s, ks in zip(shapein, kernel.shape)]
    if mode == 'valid':
        shapeout = [s - ks + 1 for s, ks in zip(shapein, kernel.shape)]
    if mode == 'same':
        shapeout = shapein
    # reverse kernel
    s = (slice(None, None, -1), ) * kernel.ndim
    rkernel = kernel[s]
    def matvec(x):
        return convolve(x, kernel, mode=mode)
    def rmatvec(x):
        if mode == 'full':
            rmode = 'valid'
        elif mode == 'valid':
            rmode = 'full'
        elif mode == 'same':
            rmode = 'same'
        return convolve(x, rkernel, mode=rmode)
    return ndoperator(shapein, shapeout, matvec, rmatvec, dtype=kernel.dtype)

def mask(mask, dtype=np.float64, copy_array=False, remove_nan=False):
    "Masking as a LinearOperator"
    shapein = mask.shape
    shapeout = mask.shape
    # make a copy to be sure mask does not change
    op_mask = copy(1 - mask)
    def matvec(x):
        if copy_array:
            y = copy(x)
        else:
            y = x
        x *= op_mask
        if remove_nan:
            x[np.isnan(x)] = 0.
        return y

    return ndoperator(shapein, shapeout, matvec, matvec, dtype=dtype)

def decimate(mask, dtype=np.float64):
    "Masking as a LinearOperator"
    shapein = mask.shape
    shapeout = np.sum(mask == False)
    def matvec(x):
        return x[mask==False]
    def rmatvec(x):
        y = np.zeros(shapein, dtype=dtype)
        y[mask==False] = x
        return y
    return ndoperator(shapein, shapeout, matvec, rmatvec, dtype=dtype)

def diff(shapein, axis=-1, dtype=np.float64):
    shapeout = np.asarray(shapein)
    shapeout[axis] -= 1
    shapetmp = list(shapeout)
    shapetmp[axis] += 2
    tmp = np.zeros(shapetmp)
    s = [slice(None),] * len(shapein)
    s[axis] = slice(1, -1)
    def matvec(x):
        return np.diff(x, axis=axis)
    def rmatvec(x):
        tmp[s] = x
        return - np.diff(tmp, axis=axis)
    return ndoperator(shapein, shapeout, matvec, rmatvec, dtype=dtype)

def binning(shapein, factor, axis=-1, dtype=np.float64):
    shapeout = np.asarray(copy(shapein))
    shapeout[axis] /= factor
    def matvec(x):
        return bin(x, factor, axis=axis)
    def rmatvec(x):
        return replicate(x, factor, axis=axis)
    return ndoperator(shapein, shapeout, matvec=matvec, rmatvec=rmatvec, 
                      dtype=dtype)

def bin(arr, factor, axis=-1):
    shapeout = np.asarray(arr.shape)
    shapeout[axis] /= factor
    outarr = np.zeros(shapeout)
    s0 = [slice(None),] * arr.ndim
    s1 = [slice(None),] * arr.ndim
    for i in xrange(arr.shape[axis]):
        s0[axis] = i
        s1[axis] = np.floor(i / factor)
        outarr[s1] += arr[s0]
    return outarr

def replicate(arr, factor, axis=-1):
    shapeout = np.asarray(arr.shape)
    shapeout[axis] *= factor
    outarr = np.zeros(shapeout)
    s0 = [slice(None),] * arr.ndim
    s1 = [slice(None),] * arr.ndim
    for i in xrange(shapeout[axis]):
        s0[axis] = i
        s1[axis] = np.floor(i / factor)
        outarr[s0] = arr[s1]
    return outarr

def axis_mul(shapein, vect, axis=-1, dtype=np.float64):
    shapeout = shapein
    def matvec(x):
        y = np.empty(x.shape)
        s = [slice(None), ] * x.ndim
        for i in xrange(x.shape[axis]):
            s[axis] = i
            y[s] = x[s] * vect
        return y
    return ndoperator(shapein, shapeout, matvec=matvec, rmatvec=matvec,
                      dtype=dtype)

def mul(shapein, num, dtype=np.float64):
    if not np.isscalar(num):
        raise ValueError('mul expect a scalar as input')
    def matvec(x):
        y = num * x
        return y
    return ndoperator(shapein, shapein, matvec=matvec, rmatvec=matvec,
                      dtype=dtype)
