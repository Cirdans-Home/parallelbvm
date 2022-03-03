import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import LinearOperator, spsolve
from utilities import dropcols_coo, vsolve
import time

class bvmethod:
    """ A class containing all the information for a given BVM """
    A = None
    B = None
    n = None
    k = None
    nu = None
    ro = None
    si = None
    formula = None
    ptype = None

    def info(self):
        """ This function prints out the information about the method """
        print("BVM Formula: "+self.formula)
        print(" Using "+str(self.k)+" steps")
        print(" ρ = "+str(self.ro.transpose()))
        print(" σ = "+str(self.si.transpose()))
        print(" ν = "+str(self.nu))
        print(" Over "+str(self.n)+" time intervals")
        print(" Preconditioner info:")
        print("     Type of structured approximation: "+str(self.ptype))

    def mab(self,type,k,n):
        """ This function creates the A and B matrices for the construction of
        the BVM.

        :param type: BVM formula "TOM", "GBDF", "GAM"
        :param k: Degree of the formula
        :param n: Number of time steps

        """
        if type.upper() == "TOM":
            print("Building TOM matrices")
        elif type.upper() == "GBDF":
            self.formula = "GBDF"
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
            self.ro = ro
            self.si = si
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
                    b_irow[row] = i
                    b_icol[row] = n+j
                    b_ival[row] = si[col]
                    row = row + 1
                    col = col + 1
            self.A = coo_matrix((a_ival,(a_irow,a_icol)),(n,n+1))
            self.B = coo_matrix((b_ival,(b_irow,b_icol)),(n,n+1))
        elif type.upper() == "GAM":
            print("Building GAM matrices")
        else:
            raise NameError('Unknown formula')

        self.n = n
        self.k = k
        self.nu = nu

    def buildlinop(self,J,T,t0,E=None):
        """ This function build the linear operator
        :math:`M= A \otimes E - h\, B \otimes J.`

        :param J: Jacobian of the system to integrate
        :param T: Final time of integration
        :param t0: Initial integration
        :param E: Mass matrix, default value is the identity matrix
        :type E: optional
        :return M: LinearOperator implementing the matrix-vector product

        """

        tic = time.perf_counter()
        A = dropcols_coo(self.A,0)
        B = dropcols_coo(self.B,0)

        s = A.shape[0] # Number of time step
        m = J.shape[0] # Size of the Jacobian
        if E is None:
            E = eye(m,format="csr",dtype=float)
        h = (T-t0)/s   # Integration step

        def mv(v):
            """ Implementation of the matrix-vector product without building the
            large Kronecker product matrix.
            """
            v = v.reshape((m,s),order='F')
            y = E*v*A.transpose() - h*J*v*B.transpose()
            y = y.reshape(m*s, order='F')
            return y

        M = LinearOperator((m*s,m*s),matvec=mv,dtype=float)
        toc = time.perf_counter()
        print(" Linear operator building time: "+"{:e}".format(toc-tic))

        return M

    def buildprecond(self,J,T,t0,ptype,E=None):
        """ This function build the linear operator
        :math:`P= \operatorname{approx}(A) \otimes E - h\, \operatorname{approx}(B) \otimes J.`

        :param J: Jacobian of the system to integrate
        :param T: Final time of integration
        :param t0: Initial integration
        :param ptype: Type of structured preconditioner to build
        :param E: Mass matrix, default value is the identity matrix
        :type E: optional
        :return P: LinearOperator implementing the preconditioner

        """

        tic = time.perf_counter()
        A = dropcols_coo(self.A,0)
        B = dropcols_coo(self.B,0)

        s = A.shape[0] # Number of time step
        m = J.shape[0] # Size of the Jacobian
        if E is None:
            E = eye(m,format="csr",dtype=float)
        h = (T-t0)/s   # Integration step

        if ptype is None:
            ptype = preconditioner()

        # Compute the eigenvalues of the Circulant parts:
        if ptype.upper() == "STRANG":
            self.ptype="Strang"
            # Compute the ϕ values
            t = np.zeros((s,1))
            t[np.arange(0,self.nu+1,1)] = self.ro[np.arange(self.nu,-1,-1)]
            t[np.arange(s-self.k+self.nu,self.n,1)] = self.ro[np.arange(self.nu+1,self.k+1,1)]
            phi = np.fft.fft(t,axis=0)
            # Compute the ψ values
            t[np.arange(0,self.nu+1,1)] =  self.si[np.arange(self.nu,-1,-1)]
            t[np.arange(s-self.k+self.nu,self.n,1)] = self.si[np.arange(self.nu+1,self.k+1,1)]
            psi = np.fft.fft(t,axis=0)
        else:
            raise NameError("Unknown Circulant Approximation")

        def mv(v):
            """ Implementation of the application of the preconditioner, this is
            the core of the parallel part of the algorithm.
            """
            v = v.reshape((m,s),order='F')
            v = np.fft.fft(v.transpose()).transpose()
            w = np.zeros((m,s),dtype="complex")
            for i in np.arange(s):
                T = E.multiply(phi[i]) + J.multiply(- h*psi[i])
                w[:,i] = spsolve(T,v[:,i])
            v = np.fft.ifft(v)
            y = v.reshape(m*s, order='F')
            return y.real

        P = LinearOperator((m*s,m*s),matvec=mv,dtype=float)
        return P

def rosi_gbdf(k,j):
    """ Builds the ρ and σ polynomials for a generalized BDF formula with

    :param k: steps
    :param j: initial conditions
    :return ro: ρ polynomial
    :return si: σ polynomial
    """
    b = np.zeros((k+1,1),dtype=float)
    b[1] = 1
    ro = vsolve(np.linspace(-j,k-j,num=k+1),b)
    si = np.zeros((k+1,1),dtype=float)
    si[j] = 1
    return ro,si
