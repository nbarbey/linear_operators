"""Definition of useful ndimensional linear operators"""
import numpy as np
from copy import copy
import warnings
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

    def __call__(self, x):
        """
        Calling an NDOperator is a short-cut for calling ndmatvec operation,
        i.e. applying the linear operation on a ndarray or subclass.
        """
        return self.ndmatvec(x)

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

class Fft(NDOperator):
    def __init__(self, shapein, n=None, axis=-1, **kwargs):
        if n > shapein[axis]:
            raise NotImplemented("The case n larger than shapein is not implemented.")
        self.n = n
        self.axis = axis
        if n is None:
            shapeout = shapein
        else:
            shapeout = np.asarray(copy(shapein))
            shapeout[axis] = n
            shapeout = tuple(shapeout)
        # normalization
        if n is None:
            self.norm = np.sqrt(shapein[axis])
        else:
            self.norm = np.sqrt(n)
        # direct
        matvec = lambda x: np.fft.fft(x, n=n, axis=axis) / self.norm
        # transpose
        # note: ifft is normalized by 1 / n by default
        # so you need to multiply by norm instead of dividing !
        if n is not None:
            def rmatvec(x):
                out = np.zeros(shapein, x.dtype)
                t = np.fft.ifft(x, axis=axis) * self.norm
                s = [slice(None) ,] * out.ndim
                s[axis] = slice(n)
                warnings.filterwarnings("ignore", category=np.ComplexWarning)
                out[s] = t
                warnings.resetwarnings()
                return out
        else:
            rmatvec = lambda x: np.fft.ifft(x, n=n, axis=axis) * self.norm
        NDOperator.__init__(self, shapein, shapeout, matvec, rmatvec, **kwargs)

class Fft2(NDOperator):
    def __init__(self, shapein, s=None, axes=(-2, -1), **kwargs):
        for a in axes:
            if s is not None:
                if s[a] > shapein[a]:
                    raise NotImplemented("The case s larger than shapein is not implemented.")
        self.s = s
        self.axes = axes
        if len(shapein) != 2:
            raise ValueError('Error expected 2 dimensional shape')
        # shapeout
        if s is None:
            shapeout = shapein
        else:
            shapeout = list(shapein)
            for a, si in zip(axes, s):
                shapeout[a] = si
        # normalization
        if s is None:
            self.norm = np.sqrt(np.prod([shapein[a] for a in axes]))
        else:
            self.norm = np.sqrt(np.prod(s))
        # matvec
        matvec = lambda x: np.fft.fft2(x, s=s, axes=axes) / self.norm
        # rmatvec
        if s is not None:
            def rmatvec(x):
                out = np.zeros(shapein, x.dtype)
                t = np.fft.ifft2(x, axes=axes) * self.norm
                sl = [slice(None), ] * out.ndim
                for si, a in zip(s, axes):
                    sl[a] = slice(si)
                warnings.filterwarnings("ignore", category=np.ComplexWarning)
                out[sl] = t
                warnings.resetwarnings()
                return out
        else:
            rmatvec = lambda x: np.fft.ifft2(x, s=s, axes=axes) * self.norm
        NDOperator.__init__(self, shapein, shapeout, matvec, rmatvec, **kwargs)

class Fftn(NDOperator):
    def __init__(self, shapein, s=None, axes=None, **kwargs):
        self.s = s
        if axes is None:
            self.axes = range(- len(shapein), 0)
        else:
            self.axes = axes
        for a in self.axes:
            if s is not None:
                if s[a] > shapein[a]:
                    raise NotImplemented("The case s larger than shapein is not implemented.")
        # shapeout
        if s is None:
            shapeout = shapein
        else:
            shapeout = list(shapein)
            for a, si in zip(self.axes, s):
                shapeout[a] = si
        # normalization
        if s is None:
            self.norm = np.sqrt(np.prod([shapein[a] for a in self.axes]))
        else:
            self.norm = np.sqrt(np.prod(s))
        # matvec
        matvec = lambda x: np.fft.fftn(x, s=s, axes=self.axes) / self.norm
        # rmatvec
        if s is not None:
            def rmatvec(x):
                out = np.zeros(shapein, x.dtype)
                t = np.fft.ifftn(x, axes=self.axes) * self.norm
                sl = [slice(None), ] * out.ndim
                for si, a in zip(s, self.axes):
                    sl[a] = slice(si)
                warnings.filterwarnings("ignore", category=np.ComplexWarning)
                out[sl] = t
                warnings.resetwarnings()
                return out
        else:
            rmatvec = lambda x: np.fft.ifftn(x, s=s, axes=self.axes) * self.norm
        NDOperator.__init__(self, shapein, shapeout, matvec, rmatvec, **kwargs)

class Convolve(NDOperator):
    def __init__(self, shapein, kernel, mode="full", fft=False, **kwargs):
        if fft:
            from scipy.signal import fftconvolve as convolve
            # check kernel shape parity
            if np.any(np.asarray(kernel.shape) % 2 != 1):
                raise ValueError("Kernels with non-even shapes are not handled for now.")
        else:
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

class ConvolveNDImage(NDOperator):
    def __init__(self, shapein, kernel, mode='reflect', cval=0.0,
                 origin=0, **kwargs):
        """
        Generate a convolution operator wrapping
        scipy.ndimage.convolve function.
        The kernel is reversed for the transposition.
        Note that kernel with even shapes are not handled.
        """
        if kernel.ndim == 1:
            from scipy.ndimage import convolve1d as convolve
        else:
            from scipy.ndimage import convolve

        self.kernel = kernel
        self.mode = mode
        self.cval = cval
        self.origin = origin

        # check kernel shape parity
        if np.any(np.asarray(self.kernel.shape) % 2 != 1):
            ValueError("kernel should have even shape.")

        # reverse kernel
        s = (slice(None, None, -1), ) * kernel.ndim
        self.rkernel = kernel[s]

        shapeout = shapein

        matvec = lambda x: convolve(x, self.kernel, mode=mode, cval=cval,
                                    origin=origin)
        rmatvec = lambda x: convolve(x, self.rkernel, mode=mode, cval=cval,
                                     origin=origin)
        NDOperator.__init__(self, shapein, shapeout, matvec, rmatvec, **kwargs)

class Fftw3(NDOperator):
    def __init__(self, shapein, **kwargs):
        import fftw3
        from multiprocessing import cpu_count
        self.n_threads = cpu_count()
        shapeout = shapein

        dtype = kwargs.get('dtype', np.float64)
        dtypein = kwargs.get('dtypein', dtype)
        dtypeout = kwargs.get('dtypeout', dtype)

        # normalize if required
        self.norm = 1.
        if dtypein == np.float64 and dtypeout == np.complex128:
            self.norm = np.sqrt(np.prod(shapein))

        self.inarray = np.zeros(shapein, dtype=dtypein)
        self.outarray = np.zeros(shapeout, dtype=dtypeout)
        self.rinarray = np.zeros(shapein, dtype=dtypeout)
        self.routarray = np.zeros(shapeout, dtype=dtypein)
        if dtype == np.complex128 or dtypein == np.complex128 or dtypeout == np.complex128:
            self.plan = fftw3.Plan(inarray=self.inarray,
                                   outarray=self.outarray,
                                   direction='forward',
                                   nthreads=self.n_threads)
            self.rplan = fftw3.Plan(inarray=self.rinarray,
                                    outarray=self.routarray,
                                    direction='backward',
                                    nthreads=self.n_threads)
        else:
            realtypes = ('halfcomplex r2c','halfcomplex r2c')
            rrealtypes = ('halfcomplex c2r','halfcomplex c2r')
            self.plan = fftw3.Plan(inarray=self.inarray,
                                   outarray=self.outarray,
                                   direction='forward',
                                   realtypes=realtypes,
                                   nthreads=self.n_threads)
            self.rplan = fftw3.Plan(inarray=self.rinarray,
                                    outarray=self.routarray,
                                    realtypes=rrealtypes,
                                    direction='backward',
                                    nthreads=self.n_threads)
        def matvec(x):
            self.inarray[:] = x[:]
            self.plan()
            return self.outarray / self.norm
        def rmatvec(x):
            self.rinarray[:] = x[:]
            self.rplan()
            return self.routarray / self.norm
        NDOperator.__init__(self, shapein, shapeout, matvec, rmatvec, **kwargs)

class ConvolveFftw3(NDOperator):
    def __init__(self, shapein, kernel, n_threads=None, **kwargs):
        import fftw3
        from multiprocessing import cpu_count
        if n_threads is None:
            self.n_threads = cpu_count()
        else:
            self.n_threads = n_threads
        self.kernel = kernel
        # reverse kernel
        #s = (slice(None, None, -1), ) * kernel.ndim
        #self.rkernel = kernel[s]
        # shapes
        shapeout = shapein
        self.fullshape = np.array(shapein) + np.array(kernel.shape) - 1
        # normalize
        self.norm = float(np.prod(self.fullshape))
        # plans
        self.inarray = np.zeros(self.fullshape, dtype=kernel.dtype)
        self.outarray = np.zeros(self.fullshape, dtype=np.complex128)
        self.rinarray = np.zeros(self.fullshape, dtype=np.complex128)
        self.routarray = np.zeros(self.fullshape, dtype=kernel.dtype)
        self.plan = fftw3.Plan(inarray=self.inarray,
                               outarray=self.outarray,
                               direction='forward',
                               nthreads=self.n_threads)
        self.rplan = fftw3.Plan(inarray=self.rinarray,
                                outarray=self.routarray,
                                direction='backward',
                                nthreads=self.n_threads)
        # for slicing
        sk = [slice(0, shapei) for shapei in kernel.shape]
        self.si = [slice(0, shapei) for shapei in shapein]
        self.so = [slice(0, shapei) for shapei in shapeout]
        # fft transform of kernel
        self.padded_kernel = np.zeros(shapein, dtype=kernel.dtype)
        self.padded_kernel[sk] = kernel
        self.fft_kernel = copy(self.fft(self.padded_kernel))
        # fft transform of rkernel
        # reverse kernel
        s = (slice(None, None, -1), ) * kernel.ndim
        self.rkernel = kernel[s]
        self.rpadded_kernel = np.zeros(shapein, dtype=self.rkernel.dtype)
        self.rpadded_kernel[sk] = self.rkernel
        self.rfft_kernel = copy(self.fft(self.rpadded_kernel))
        # matvec
        def matvec(x):
            return self._centered(self.convolve(x, self.fft_kernel), shapeout) / self.norm
        # rmatvec
        def rmatvec(x):
            return self._centered(self.convolve(x, self.rfft_kernel), shapein) / self.norm
        NDOperator.__init__(self, shapein, shapeout, matvec, rmatvec, **kwargs)

    def fft(self, arr):
        self.inarray[:] = 0.
        self.outarray[:] = 0.
        self.inarray[self.si] = arr
        self.plan()
        return self.outarray

    def ifft(self, arr):
        self.rinarray[:] = 0.
        self.routarray[:] = 0.
        self.rinarray[self.si] = arr[self.si]
        self.rplan()
        return self.routarray

    def convolve(self, arr, fft_kernel):
        return self.ifft(self.fft(arr) * fft_kernel)

    def _centered(self, arr, newsize):
        # Return the center newsize portion of the array.
        newsize = np.asarray(newsize)
        currsize = np.array(arr.shape)
        startind = (currsize - newsize) / 2
        endind = startind + newsize
        myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
        return arr[tuple(myslice)]

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
        shapeout[axis] = np.ceil(shapeout[axis] / float(factor))

        matvec = lambda x: self.bin(x, factor, axis=axis)
        rmatvec = lambda x: self.replicate(x, factor, axis=axis)
        NDOperator.__init__(self, shapein, shapeout, matvec, rmatvec, **kwargs)

    def bin(self, arr, factor, axis=-1):
        outarr = np.zeros(self.shapeout)
        s0 = [slice(None),] * arr.ndim
        s1 = [slice(None),] * arr.ndim
        for i in xrange(arr.shape[axis]):
            s0[axis] = i
            s1[axis] = np.floor(i / factor)
            outarr[s1] += arr[s0]
        return outarr

    def replicate(self, arr, factor, axis=-1):
        outarr = np.zeros(self.shapein)
        s0 = [slice(None),] * arr.ndim
        s1 = [slice(None),] * arr.ndim
        for i in xrange(self.shapein[axis]):
            s0[axis] = i
            s1[axis] = np.floor(i / factor)
            outarr[s0] = arr[s1]
        return outarr

class NDSlice(NDOperator):
    def __init__(self, shapein, slices, **kwargs):
        # compute shapeout (np.max handles s.step == None case)
        shapeout = []
        for a, s in enumerate(slices):
            if s.stop is None:
                stop = shapein[a] -1
            else:
                stop = s.stop
            start = np.max((s.start, 0))
            step = np.max((1, s.step))
            shapeout.append(int(np.ceil((stop - start + 1) / float(step))))
        shapeout = tuple(shapeout)
        #shapeout = [int(np.floor((s.stop - s.start) / np.max(1, s.step))) for s in slices]
        def matvec(x):
            return x[slices]
        def rmatvec(x):
            out = np.zeros(shapein)
            out[slices] = x
            return out
        NDOperator.__init__(self, shapein, shapeout, matvec, rmatvec=rmatvec, **kwargs)

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

def fft(shapein, n=None, axis=-1, **kwargs):
    return Fft(shapein, n=n, axis=axis, **kwargs)

def fft2(shapein, s=None, axes=(-2, -1), **kwargs):
    return Fft2(shapein, s=s, axes=axes, **kwargs)

def fftn(shapein, s=None, axes=None, **kwargs):
    return Fftn(shapein, s=s, axes=axes, **kwargs)

def convolve(shapein, kernel, mode="full", fft=False, **kwargs):
    return Convolve(shapein, kernel, mode=mode, fft=fft, **kwargs)

def convolve_ndimage(shapein, kernel, mode="reflect", cval=0.0, origin=0,
                     **kwargs):
    return ConvolveNDImage(shapein, kernel, mode=mode, cval=cval,
                           origin=origin, **kwargs)
def convolve_fftw3(shapein, kernel, **kwargs):
    return ConvolveFftw3(shapein, kernel, **kwargs)

def diff(shapein, axis=-1, **kwargs):
    return Diff(shapein, axis=axis, **kwargs)

def binning(shapein, factor, axis=-1, **kwargs):
    return Binning(shapein, factor, axis=axis, **kwargs)

def ndslice(shapein, slices, **kwargs):
    return NDSlice(shapein, slices, **kwargs)
