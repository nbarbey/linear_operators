""" Wrap pywt transform into LinearOperators """
import numpy as np
import pywt
import lo
from copy import copy

def wavedec(shapein, wavelet, mode='sym', level=None, dtype=np.float64):
    a = np.zeros(shapein)
    b = pywt.wavedec(a, wavelet, mode=mode, level=level)
    def matvec(x):
        dec_list = pywt.wavedec(x, wavelet, mode=mode, level=level)
        return np.concatenate(dec_list)
    def rmatvec(x):
        x_list = []
        count = 0
        for el in b:
            n_el = np.asarray(el).size
            x_list.append(np.array(x[count:count+n_el]))
            count += n_el
        return pywt.waverec(x_list, wavelet, mode=mode)
    shapeout = matvec(a).size
    return lo.LinearOperator((shapeout, shapein), matvec=matvec, rmatvec=rmatvec,
                             dtype=dtype)

def wavelet2(shapein, wavelet, mode='zpd', level=None, dtype=np.float64):
    """
    2d wavelet decomposition / reconstruction as a NDOperator
    """
    a = np.zeros(shapein)
    b = pywt.wavedec2(a, wavelet, mode=mode, level=level)
    shapeout = coefs2array(b).shape
    def matvec(x):
        coefs = pywt.wavedec2(x, wavelet, mode=mode, level=level)
        return coefs2array(coefs)
    def rmatvec(x):
        coefs = array2coefs(x, b)
        return pywt.waverec2(coefs, wavelet, mode=mode)
    return lo.NDOperator(shapein, shapeout, matvec, rmatvec, dtype=dtype)

def coefs2array(coefs):
    out = coefs[0]
    for scale in coefs[1:]:
        out = np.vstack((np.hstack((out, scale[0])), np.hstack((scale[1], scale[2]))))
    return out

def array2coefs(a, l):
    ilim = [0, l[0].shape[0]]
    jlim = [0, l[0].shape[1]]
    coefs = [a[ilim[0]:ilim[1], jlim[0]:jlim[1]], ]
    for i in xrange(1, len(l)):
        ilim.append(ilim[-1] + l[i][0].shape[0])
        jlim.append(jlim[-1] + l[i][0].shape[1])
        scale = (a[0:ilim[-2], jlim[-2]:jlim[-1]],
                 a[ilim[-2]:ilim[-1], 0:jlim[-2]],
                 a[ilim[-2]:ilim[-1], jlim[-2]:jlim[-1]])
        coefs.append(scale)
    return coefs
