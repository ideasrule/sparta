from constants import USE_GPU

if USE_GPU:
    from cupy import *
    from cupyx import scipy
else:
    from numpy import *
    import scipy

def cpu(arr):
    try:
        return arr.get()
    except:
        return arr
