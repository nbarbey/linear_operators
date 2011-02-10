# import wrappers if the corresponding module exist
try:
    import pywt
except ImportError:
    pass

if 'pywt' in locals():
    from pywt_lo import *
    del pywt

try:
    import fht
except ImportError:
    pass

if 'fht' in locals():
    from fht_lo import fht
