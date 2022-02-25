import numpy as np

def vsolve(x,b):
    # This function solves the Vandermonde linear system W(x)f = b;
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

def rosi_gbdf(k,j):
    # Builds the ρ and σ polynomials for a generalized BDF formula with:
    # k steps
    # j initial conditions
    # Remark: If J=K we get an Initial Value Method
    #         If J=fix(k/2)+1 we get A-Stable GBDF
    b = np.zeros((k+1,1),dtype=float)
    b[1] = 1
    ro = vsolve(np.linspace(-j,k-j,num=k+1),b)
    si = np.zeros((k+1,1),dtype=float)
    si[j] = 1
    return ro,si


def mab(type,k,n):
    # This function creates the A and B matrices
    # for the construction of the BVM scheme.
    if type.upper() == "TOM":
        print("Building TOM matrices")
    elif type.upper() == "GBDF":
        print("Building GBDF matrices")
    elif type.upper() == "GAM":
        print("Building GAM matrices")
    else:
        raise NameError('Unknown formula')
