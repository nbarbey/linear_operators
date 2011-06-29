"""Wrap fast hadamard transform (fht) into LinearOperator instances"""
import numpy as np
import fht as fht_mod
from ..ndoperators import ndoperator

def fht(shapein, **kargs):
    """
    Fast Hadamard transform LinearOperator
    """
    def matvec(arr):
        return fht_mod.fht(arr, **kargs)
    return ndoperator(shapein, shapein, matvec=matvec, rmatvec=matvec, 
                      dtype=kargs.get("dtype", np.float64))
