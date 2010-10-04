"""
Use cudamat to perform faster matrix vector multiplications
"""
import cudamat
from cudamat import CUDAMatrix
import numpy as np
import lo

def cudamat_aslinearoperator(a):
    if isinstance(a, np.ndarray):
        a_gpu = CUDAMatrix(a)
    elif isinstance(a, CUDAMatrix):
        a_gpu = a
    else:
        raise ValueError('Expected CUDAMatrix or ndarray')
    # define linear operator
    def matvec(x):
        if isinstance(x, np.ndarray):
            x.resize((x.size, 1))
            x_gpu = CUDAMatrix(x)
            return cudamat.dot(a_gpu, x_gpu).asarray()
        elif isinstance(x, CUDAMatrix):
            x_gpu = x
            return cudamat.dot(a_gpu, x_gpu)
        else:
            raise ValueError('Expected CUDAMatrix or ndarray')
    def rmatvec(x):
        if isinstance(x, np.ndarray):
            x.resize((x.size, 1))
            x_gpu = CUDAMatrix(x)
            return cudamat.dot(a_gpu.transpose(), x_gpu).asarray()
        elif isinstance(x, CUDAMatrix):
            x_gpu = x
            return cudamat.dot(a_gpu.transpose(), x_gpu)
        else:
            raise ValueError('Expected CUDAMatrix or ndarray')
    return lo.LinearOperator(a.shape, matvec, rmatvec, dtype=a.dtype)
