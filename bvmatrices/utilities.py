import numpy as np
from scipy import sparse

def vsolve(x,b):
    """ This function solves the Vandermonde linear system :math:`W(x)f = b`.
    It is a service function needed to compute the ρ and σ polynomials.

    :param x: vector of the variables generating the Vandermonde system
    :param b: right-hand side of the system
    :return f: solution of the linear system
    """
    f = b
    n = x.size - 1
    for k in np.arange(0,n):
        for i in np.arange(n,k,-1):
            f[i] = f[i] - x[k]*f[i-1]
    for k in np.arange(n-1,-1,-1):
        for i in np.arange(k+1,n+1):
            f[i] = f[i]/(x[i]-x[i-k-1])
        for i in np.arange(k,n):
            f[i] = f[i] - f[i+1]
    return f

def dropcols_coo(C, idx_to_drop):
    """ Drops columns from matrices stored in COO format. Result is returned in
    CSR format.

    :param C: Matrix in COO format.
    :param idx_to_drop: List of columns to be dropped.
    :return C: Matrix in CSR format with dropped columns.
    """
    idx_to_drop = np.unique(idx_to_drop)
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C.tocsr()
