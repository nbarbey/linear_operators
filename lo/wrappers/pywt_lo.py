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

#def wavedec2(shapein, wavelet, mode='sym', level=None, dtype=np.float64):
#    a = np.zeros(shapein)
#    b = pywt.wavedec2(a, wavelet, mode=mode, level=level)
#    def matvec(x):
#        y = x.reshape(shapein)
#        input_list = pywt.wavedec2(y, wavelet, mode=mode, level=level)
#        output_array = input_list[0]
#        for element in input_list[1:]:
#            output_array = np.hstack((output_array, element[0]))
#            tmp = np.hstack((element[1], element[2]))
#            output_array = np.vstack((output_array, tmp))
#        return output_array
#    def rmatvec(x):
#        input_array = x.reshape(shapein)
#        output_list = list()
#        for i in xrange(len(b) - 1):
#            tmp1, tmp2 = np.vsplit(input_array, 2)
#            input_array, tmp3 = np.hsplit(tmp1, 2)
#            tmp4, tmp5 = np.hsplit(tmp2, 2)
#            output_list.append(tuple((tmp3, tmp4, tmp5)))
#        output_list.append(input_array)
#        output_list.reverse()
#        return pywt.waverec2(output_list, wavelet, mode=mode)
#    shapeout = matvec(a).size
#"    return lo.ndoperator(shapeout, shapein, matvec=matvec, rmatvec=rmatvec,
#                         dtype=dtype)

def wavelet2(shapein, wavelet, mode='zpd', level=None, dtype=np.float64):
    a = np.zeros(shapein)
    b = pywt.wavedec2(a, wavelet, mode=mode, level=level)
    def matvec(x):
        xnd = x.reshape(shapein)
        yl = pywt.wavedec2(xnd, wavelet, mode=mode, level=level)
        y = yl[0].flatten()
        for el in yl[1:]:
            y = np.concatenate((y, np.concatenate(el).flatten()))
        return y
    def rmatvec(x):
        iinf = 0
        isup = b[0].size
        yl = [x[iinf:isup].reshape(b[0].shape), ]
        for i in xrange(1, len(b)):
            tmp = list()
            for j in xrange(3):
                iinf = copy(isup)
                isup = iinf + b[i][j].size
                tmp.append(x[iinf:isup].reshape(b[i][j].shape))
            yl.append(tmp)
        return pywt.waverec2(yl, wavelet, mode=mode)[:a.shape[0], :a.shape[1]].flatten()
    shape = (matvec(a).size, np.prod(shapein))
    return lo.LinearOperator(shape, matvec=matvec, rmatvec=rmatvec)
