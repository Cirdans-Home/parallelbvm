import numpy as np
from scipy.sparse import coo_matrix

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
        nu = int(np.fix((k+2)/2))
        a_irow = np.zeros(shape=(k+1)*n,dtype=int)
        a_icol = np.zeros(shape=(k+1)*n,dtype=int)
        a_ival = np.zeros(shape=(k+1)*n,dtype=float)
        b_irow = np.zeros(shape=(k+1)*n,dtype=int)
        b_icol = np.zeros(shape=(k+1)*n,dtype=int)
        b_ival = np.zeros(shape=(k+1)*n,dtype=float)
        row = 0
        # We start with the entries relative to boundary conditions on the left
        for i in np.arange(0,nu):
            [ro,si] = rosi_gbdf(k,i+1)
            for j in np.arange(0,k+1):
                a_irow[row] = i
                a_icol[row] = j
                a_ival[row] = ro[j]
                b_irow[row] = i
                b_icol[row] = j
                b_ival[row] = si[j]
                row = row + 1
        # Then we populate the central part of the matrix
        for i in np.arange(nu,n-(k-nu)):
            col = 0
            for j in np.arange(-nu,k-nu+1):
                a_irow[row] = i
                a_icol[row] = i+1+j
                a_ival[row] = ro[col]
                b_irow[row] = i
                b_icol[row] = i+1+j
                b_ival[row] = si[col]
                row = row + 1
                col = col + 1
        # Finally we put in place boundary conditions on the right
        l = nu
        for i in np.arange(n-(k-nu),n):
            l = l + 1
            [ro,si] = rosi_gbdf(k,l)
            col = 0
            for j in np.arange(-k,1):
                a_irow[row] = i
                a_icol[row] = n+j
                a_ival[row] = ro[col]
                print("3] a("+str(a_irow[row])+","+str(a_icol[row])+")="+str(a_ival[row]))
                b_irow[row] = i
                b_icol[row] = n+j
                b_ival[row] = si[col]
                row = row + 1
                col = col + 1
        A = coo_matrix((a_ival,(a_irow,a_icol)),(n,n+1))
        B = coo_matrix((b_ival,(b_irow,b_icol)),(n,n+1))
    elif type.upper() == "GAM":
        print("Building GAM matrices")
    else:
        raise NameError('Unknown formula')

    return A,B
