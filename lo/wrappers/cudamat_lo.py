"""
Use cudamat to perform faster matrix vector multiplications
"""
import cudamat
import numpy as np
import lo

def aslinearoperator(a):
    if isinstance(a, np.ndarray):
        a_gpu = cudamat.CUDAMatrix(a)
    elif isinstance(a, cudamat.CUDAMatrix):
        a_gpu = a
    else:
        raise ValueError('Expected CUDAMatrix or ndarray')
    # define linear operator
    matvec = lambda x: cudamat.dot(a, x)
    rmatvec = lambda x: cudamat.dot(a.transpose(), x)
    return lo.LinearOperator(a.shape, matvec, rmatvec, dtype=a.dtype)
