import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import LinearOperator, spsolve
from scipy.io import savemat
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

    def savetomatlab(self):
        """ Save the building blocks of the method in MATLAB format """
        savemat(self.formula+".mat",{'A':self.A,'B':self.B,'n':self.n,
            'k':self.k,'ro':self.ro,'si':self.si},oned_as="column")

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

    def buildlinop(self,J,T,t0,g,u0,E=None):
        """ This function build the linear operator
        :math:`M= A \otimes E - h\, B \otimes J.` and the right-hand side for
        the liner system.

        :param J: Jacobian of the system to integrate
        :param T: Final time of integration
        :param t0: Initial integration
        :param g: Right-Hand side, could be either a vector or a function
        :param u0: Initial condition
        :param E: Mass matrix, default value is the identity matrix
        :type E: optional
        :return M: LinearOperator implementing the matrix-vector product

        """

        # First we build the rhs for which we need the A and B matrices with
        # of the whole size
        tic = time.perf_counter()
        m = J.shape[0] # Size of the Jacobian
        s = self.A.shape[0] # Number of time step
        h = (T-t0)/s   # Integration step
        if E is None:
            E = eye(m,format="csr",dtype=float)

        G = np.zeros((m,s+1))
        u0mat = np.zeros((m,s+1))
        u0mat[0:m,0] = u0[0:m]
        u0mat = E*u0mat*self.A.transpose() - h*J*u0mat*self.B.transpose()

        if callable(g):
            for i in np.arange(1,s+1):
                G[:,i] = g(t0 + i*h)
        else:
            for i in np.arange(1,s+1):
                G[:,i] = g[0:m]
        G[:,1::] = (h*G*self.B.transpose())
        G[:,0] = u0[0:m]
        G[:,1:self.k] = G[:,1:self.k] - u0mat[:,0:self.k-1]

        print(G.shape)

        G = G.reshape( m*(s+1),order='F')
        toc = time.perf_counter()
        print(" RHS building time: "+"{:e}".format(toc-tic))

        tic = time.perf_counter()
        A = dropcols_coo(self.A,0)
        B = dropcols_coo(self.B,0)

        def mv(v):
            """ Implementation of the matrix-vector product without building the
            large Kronecker product matrix.
            """
            v = v.reshape((m,s),order='F')
            y = E*v*A.transpose() - h*J*v*B.transpose()
            y = y.reshape(m*s, order='F')
            return y

        # savemat("matprod.mat",{"A": A, "B": B, "E": E, "J": J, "G":G})

        M = LinearOperator((m*s,m*s),matvec=mv,dtype=float)
        toc = time.perf_counter()
        print(" Linear operator building time: "+"{:e}".format(toc-tic))

        return M,G[m::]

    def buildrhs(self,u0,g,T,t0):
        """ This function bulds the right-hand side for the all-at-once system
        with the given BVM formula.

        :param u0: Initial condition
        :param g: source vector, it can be both a function of t or a constant
            vector
        :param T: Final time of integration
        :param t0: Initial integration
        :return rhs:
        """

        s = self.A.shape[0] # Number of time step
        m = u0.shape[0]     # Size of the Jacobian
        rhs = np.zeros(m*s)
        rhs[0:m] = u0[0:m]
        G = np.zeros((m,s))
        h = (T - t0)/s

        if callable(g):
            for i in np.arange(s):
                G[:,i] = g(t0 + i*h)
        else:
            for i in np.arange(s):
                G[:,i] = g[0:m]

        rhs = rhs + (h*G*self.B).reshape( m*s,order='F')

        return rhs.transpose()


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
            phi[0] = phi[s-1].real
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
            v = np.fft.ifft(w.transpose()).transpose()
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
