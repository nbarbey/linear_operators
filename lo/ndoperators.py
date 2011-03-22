"""Definition of useful ndimensional linear operators"""
import numpy as np
from copy import copy
from interface import LinearOperator

class NDOperator(LinearOperator):
    """Subclass of LinearOperator that handle multidimensional inputs and outputs"""
    def __init__(self, shapein, shapeout, matvec, rmatvec=None, matmat=None, rmatmat=None,
                 dtypein=None, dtypeout=None, dtype=np.float64):

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

        # rename to keep same interface as LinearOperator
        self.ndmatvec = matvec
        self.ndrmatvec = rmatvec
        self.shapein = shapein
        self.shapeout = shapeout

    @property
    def T(self):
        kwargs = {}
        if self.rmatvec is not None:
            matvec, kwargs['rmatvec'] = self.ndrmatvec, self.ndmatvec
        else:
            raise NotImplementedError('rmatvec is not defined')
        if self._matmat is not None and self._rmatmat is not None:
            kwargs['matmat'], kwargs['rmatmat'] = self._rmatmat, self._matmat
        else:
            matmat, rmatmat = None, None
        kwargs['dtype'] = getattr(self, 'dtype', None)
        kwargs['dtypein'] = getattr(self, 'dtypeout', None)
        kwargs['dtypeout'] = getattr(self, 'dtypein', None)
        shapein = self.shapeout
        shapeout = self.shapein
        return NDOperator(shapein, shapeout, matvec, **kwargs)

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

# generator
        
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

# subclasses
class NDSquare(NDOperator):
    def __init__(self, shapein, matvec, **kwargs):
        NDOperator.__init__(self, shapein, shapein, matvec, **kwargs)
    
class NDSymmetric(NDSquare):
    def __init__(self, shapein, matvec, **kwargs):
        kwargs['rmatvec'] = matvec
        kwargs['rmatmat'] = kwargs.get("matmat")
        NDSquare.__init__(self, shapein, matvec, **kwargs)

    @property
    def T(self):
        return self

class NDIdentity(NDSymmetric):
    def __init__(self, shapein, **kwargs):
        identity = lambda x: x
        NDSymmetric.__init__(self, shapein, identity, **kwargs)

    def __mul__(self, x):
        if isinstance(x, LinearOperator):
            return x
        else:
            super(NDIdentity, self).__mul__(x)

    def __rmul__(self, x):
        if isinstance(x, LinearOperator):
            return x
        else:
            super(NDIdentity, self).__mul__(x)

class NDHomothetic(NDSymmetric):
    def __init__(self, shapein, coef, **kwargs):
        self.coef = coef
        matvec = lambda x: coef * x
        NDSymmetric.__init__(self, shapein, matvec, **kwargs)
    def __repr__(self):
        s = LinearOperator.__repr__(self)
        return s[:-1] + " and coef=%f >" % self.coef

class NDDiagonal(NDSymmetric):
    def __init__(self, d, **kwargs):
        shapein = d.shape
        self.d = d
        matvec = lambda x: x * d
        NDSymmetric.__init__(self, shapein, matvec, **kwargs)
    def __repr__(self):
        s = LinearOperator.__repr__(self)
        return s[:-1] + "\n and diagonal=" + self.d.__repr__() + ">"

class NDMask(NDDiagonal):
    def __init__(self, mask, **kwargs):
        self.mask = mask.astype(np.bool)
        NDDiagonal.__init__(self, self.mask, **kwargs)

class Decimate(NDOperator):
    def __init__(self, mask, **kwargs):
        self.mask = mask.astype(np.bool)
        self.mask_inv = self.mask == False
        shapein = mask.shape
        shapeout = np.sum(self.mask_inv)
        self._in = np.zeros(shapein)
        self._out = np.zeros(shapeout)
        def matvec(x):
            self._out[:] = x[self.mask_inv]
            return self._out
        def rmatvec(x):
            self._in[self.mask_inv] = x
            return self._in
        NDOperator.__init__(self, shapein, shapeout, matvec, rmatvec, **kwargs)

class Fft2(NDOperator):
    def __init__(self, shapein, s=None, axes=(-2, -1), **kwargs):
        self.s = s
        self.axes = axes
        if len(shapein) != 2:
            raise ValueError('Error expected 2 dimensional shape')
        if s is None:
            shapeout = shapein
        else:
            shapeout = s
        matvec = lambda x: np.fft.fft2(x, s=s, axes=axes) / np.sqrt(np.prod(shapein))
        rmatvec = lambda x: np.fft.ifft2(x, s=s, axes=axes) * np.sqrt(np.prod(shapein))
        NDOperator.__init__(self, shapein, shapeout, matvec, rmatvec, **kwargs)

class Fftn(NDOperator):
    def __init__(self, shapein, s=None, axes=None, **kwargs):
        self.s = s
        self.axes = axes
        if s is None:
            shapeout = shapein
        else:
            shapeout = s
        matvec = lambda x: np.fft.fftn(x, s=s, axes=axes) / np.sqrt(np.prod(shapein))
        rmatvec = lambda x: np.fft.ifftn(x, s=s, axes=axes) * np.sqrt(np.prod(shapein))
        NDOperator.__init__(self, shapein, shapeout, matvec, rmatvec, **kwargs)

class Convolve(NDOperator):
    def __init__(self, shapein, kernel, mode="full", **kwargs):
        from scipy.signal import convolve

        self.kernel = kernel
        self.mode = mode

        # reverse kernel
        s = (slice(None, None, -1), ) * kernel.ndim
        self.rkernel = kernel[s]
        # reverse mode
        if mode == 'full':
            self.rmode = 'valid'
        elif mode == 'valid':
            self.rmode = 'full'
        elif mode == 'same':
            self.rmode = 'same'
        # shapeout
        if mode == 'full':
            shapeout = [s + ks - 1 for s, ks in zip(shapein, kernel.shape)]
        if mode == 'valid':
            shapeout = [s - ks + 1 for s, ks in zip(shapein, kernel.shape)]
        if mode == 'same':
            shapeout = shapein

        matvec = lambda x: convolve(x, self.kernel, mode=self.mode)
        rmatvec = lambda x: convolve(x, self.rkernel, mode=self.rmode)
        NDOperator.__init__(self, shapein, shapeout, matvec, rmatvec, **kwargs)

class Diff(NDOperator):
    def __init__(self, shapein, axis=-1, **kwargs):
        self.axis = axis
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
        NDOperator.__init__(self, shapein, shapeout, matvec, rmatvec, **kwargs)

class Binning(NDOperator):
    def __init__(self, shapein, factor, axis=-1, **kwargs):
        self.factor = factor
        self.axis = axis

        shapeout = np.asarray(copy(shapein))
        shapeout[axis] /= factor

        matvec = lambda x: self.bin(x, factor, axis=axis)
        rmatvec = lambda x: self.replicatebin(x, factor, axis=axis)
        NDOperator.__init__(self, shapein, shapeout, matvec, rmatvec, **kwargs)

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

# functions
def ndidentity(shapein, **kwargs):
    return NDIdentity(shapein, **kwargs)

def ndhomothetic(shapein, coef, **kwargs):
    return NDHomothetic(shapein, coef, **kwargs)

def nddiagonal(d, **kwargs):
    return NDDiagonal(d, **kwargs)

def ndmask(mask, **kwargs):
    return NDMask(mask, **kwargs)

def decimate(mask, **kwargs):
    return Decimate(mask, **kwargs)

def fft2(shapein, s=None, axes=(-2, -1), **kwargs):
    return Fft2(shapein, s=s, axes=axes, **kwargs)

def fftn(shapein, s=None, axes=None, **kwargs):
    return Fftn(shapein, s=s, axes=axes, **kwargs)

def convolve(shapein, kernel, mode="full", **kwargs):
    return Convolve(shapein, kernel, mode=mode)

def diff(shapein, axis=-1, **kwargs):
    return Diff(shapein, axis=axis, **kwargs)

def binning(shapein, factor, axis=-1, **kwargs):
    return Binning(shapein, factor, axis=axis, **kwargs)
