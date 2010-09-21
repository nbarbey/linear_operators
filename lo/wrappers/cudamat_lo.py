"""
Use cudamat to perform faster matrix vector multiplications
"""
import cudamat
import numpy as np
import lo

def aslinearoperator(a):
    from cudamat import CUDAMatrix
    if isinstance(a, np.ndarray):
        a_gpu = CUDAMatrix(a)
    elif isinstance(a, cudamat.CUDAMatrix):
        a_gpu = a
    else:
        raise ValueError('Expected CUDAMatrix or ndarray')
    # define linear operator
    def matvec(x):
        x_gpu = CUDAMatrix(x.reshape((x.size, 1)))
        return cudamat.dot(a_gpu, x_gpu).asarray()
    def rmatvec(x):
        x_gpu = CUDAMatrix(x.reshape((x.size, 1)))
        return cudamat.dot(a_gpu.transpose(), x_gpu).asarray()
    return lo.LinearOperator(a.shape, matvec, rmatvec, dtype=a.dtype)
