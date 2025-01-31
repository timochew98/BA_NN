import numpy as np
from pyamg.util.linalg import norm, approximate_spectral_radius

from pyamg.aggregation.tentative import fit_candidates

from amgqcd.low_precision.scale import *

from .relaxation import polynomial, polynomial_low_vector, make_system, richardson_prolongation_smoother

from scipy import sparse

from amgqcd.linear_solver.cg import *

from scipy.sparse.linalg import LinearOperator

from amgqcd.dirac.D_sparse import *

from scipy.sparse.linalg._isolve.utils import make_system as make_system_cg

from .setup import *

def Norm(vector):
    vector_flatten = vector.flatten()
    return np.dot(np.conjugate(vector_flatten),vector_flatten)


class TwoGridBase:
    def __init__(self, bit_precision = 5, rtol = 1e-3, numCandidates = 8) -> None:

        self.max_value = 2**(bit_precision-1) - 1

        self.bit_precision = bit_precision
        self.rtol = rtol
        self.numCandidates = numCandidates
        self.B, self.P, self.R, self.Ac = [None, None, None, None]

        self.counter_low = 0
        self.counter_high = 0
        self.counter_mid = 0

        self.relax = self.relax_high
        self.polynomial = self.polynomial_high
        self.setup_starter = SetupStarter()
        # self.calculate_aggregate()
    
    def update_A(self, A):
        self.A = A
        self.A_high = A.copy()
        self.coefficient = 1.0/approximate_spectral_radius(self.A_high)
        self.A_single = self.A.astype("complex64").astype("complex128")
    
    def update_A_single(self):
        self.A_single = self.A.astype("complex64").astype("complex128")


    def update_A_low(self):
        A_factor = scale_factor(self.A, max_value= self.max_value)
        self.A_low = np.round(self.A_high*A_factor)/A_factor

    def calculate_aggregate(self):
        pass
    
    def relax_high(self, A, x):
        self.counter_high += 2
        polynomial(A, x, np.zeros_like(x), iterations=1,
                    coefficients=[self.coefficient]) 
        
    def relax_low(self, A, x):
        self.counter_low += 2
        polynomial_low_vector(A, x, np.zeros_like(x), iterations=1,
                    coefficients=[self.coefficient], bit_precision= self.bit_precision)
        
    def polynomial_high(self,A,x,b):
        self.counter_high += 2
        polynomial(A, x, b, iterations=1,
                        coefficients=[self.coefficient])
        
    def polynomial_low(self,A,x,b):
        self.counter_low += 2
        polynomial_low_vector(A, x, b, iterations=1,
                        coefficients=[self.coefficient], bit_precision= self.bit_precision)
    

    def change_relax_method(self, low_precision, bit_precision = None):
        
        self.A_dtype = self.A.dtype
        if low_precision:

            if bit_precision is not None:
                self.bit_precision = bit_precision
                self.max_value = 2**(bit_precision-1) - 1
                self.update_A_low()

            self.relax = self.relax_low
            self.polynomial = self.polynomial_low
            self.A = self.A_low.copy().astype(self.A_dtype)


        else:
            self.relax = self.relax_high
            self.polynomial = self.polynomial_high
            self.A = self.A_high.copy().astype(self.A_dtype)

    def run_setup_phase(self, setup_name = None, **kwargs):
        self.setup_starter.run(self, setup_name = setup_name, numCandidates = self.numCandidates, **kwargs)


    def create_operators(self, B):
        self.B = B
        T_l, _ = fit_candidates(self.AggOp, B)  # step 4c   # TIMOTEO: I think that I can just reuse this code for the moment


        self.P = richardson_prolongation_smoother(self.A_high, T_l)
        self.R = np.conjugate(self.P.T).asformat(self.P.format)
        self.Ac = self.R * self.A_high * self.P


    def _solve(self, A, A_high, Ac, P, R, b, numPre = 2, numPost = 2, print_coarse_iterations = False, M = None, x = None):

        if x is None:
            x = np.zeros_like(b)

        for _ in range(numPre):
            self.polynomial(A, x, b)

        
        residual = b- A_high.dot(x)
        coarse_b = R.dot(residual)

        solution, info = cg(Ac, coarse_b, rtol=self.rtol, maxiter=10000, M = M)
        # if print_coarse_iterations:
        #     print(info)

        x2 = P.dot(solution)

        x += x2

        for _ in range(numPost):
            self.polynomial(A, x, b)

        return x

    def solve(self, b, numPre = 2, numPost = 2, print_coarse_iterations = False, M = None, x = None, single_precision = True):
        if single_precision:
            if self.A.dtype.name.startswith('complex'):
                precision = "complex64"
                double_precision = "complex128"
            else:
                precision = "float32"
                double_precision = "float64"
            return self._solve(self.A, self.A_single, self.Ac,
                        self.P, self.R, b.astype(precision).astype(double_precision),
                        numPre = numPre, numPost = numPost,
                        print_coarse_iterations = print_coarse_iterations, M = M, x = x)
        else:
            return self._solve(self.A, self.A_high, self.Ac, self.P, self.R, b,
                        numPre = numPre, numPost = numPost,
                        print_coarse_iterations = print_coarse_iterations, M = M, x = x)
    

    # def aspreconditioner(self, numPre = 2, numPost = 2, print_coarse_iterations = False, psolve_preconditioner = None, single_precision = True):
    #     def psolve(b):
    #         # print(self.A.shape)
    #         return self.solve(b, numPre = numPre, numPost = numPost,
    #                             print_coarse_iterations = print_coarse_iterations, psolve = psolve_preconditioner, single_precision = single_precision)

    #     return psolve

    def aspreconditioner(self, numPre = 2, numPost = 2, print_coarse_iterations = False, M = None, single_precision = True):
        shape = self.A.shape
        dtype = self.A.dtype

        def matvec(b):
            return self.solve(b, numPre = numPre, numPost = numPost,
                                print_coarse_iterations = print_coarse_iterations, M = M, single_precision = single_precision)

        return LinearOperator(shape, matvec, dtype=dtype)


    def mgcg(self, b, atol = 0., rtol = 9e-12, max_inner = 8, max_outer = 40,
                              numPre=4, numPost=0, print_coarse_iterations = False, single_precision = True):
        
        
        if self.A.dtype.name.startswith('complex'):
            double_precision = "complex128"
            if single_precision:
                precision = "complex64"
            else:
                precision = double_precision
        else:
            double_precision = "float64"
            if single_precision:
                precision = "float32"
            else:
                precision = double_precision

        M = self.aspreconditioner(print_coarse_iterations = print_coarse_iterations,
                                   numPre=numPre, numPost=numPost,
                                   single_precision= single_precision)
        
        bnrm2 = np.linalg.norm(b)
        atol, _ = get_atol_rtol('cg', bnrm2, atol, rtol)
  

        counter = 0 
        # print(self.B.shape, self.Ac.shape)
        x,info = cg(self.A_single, b.astype(precision).astype(double_precision), rtol=1e-1, maxiter=max_inner, M=M)
        # print("CHAU")
        x= x.astype(double_precision)
        counter+=info

        r = b - self.A_high.dot(x)
        
        for i in range(max_outer):
            y,info = cg(self.A_single, r.astype(precision).astype(double_precision), rtol=1e-1, maxiter=max_inner, M=M)
            counter+=info
            x+=y
            r = b - self.A_high.dot(x)
            rS = np.linalg.norm(r)
            if rS <atol:
                break
        return x,[counter,i + 1],rS


class TwoGridQCD(TwoGridBase):
    def __init__(self, gauge_links, Nt, Nx,Nc,Ns, m = None, size_t = 4, size_x = 4, bit_precision = 5, rtol = 1e-3) -> None:
        self.calc = CalculatorD(Nt,Nx,Nc,Ns,0)
        super().__init__(bit_precision=bit_precision, rtol=rtol)

        self.start_D_original(gauge_links)
        if m is not None:
            self.update_m(m)

        self.Nt = Nt 
        self.Nx = Nx
        self.Nc = Nc 
        self.Ns = Ns
        self.lattice_volume = Nt*Nx

        self.size_t = size_t
        self.size_x = size_x

        self.size_vector = Nc*Ns
        self.AggOp = self.calculate_aggregate(self.size_t, self.size_x, self.size_vector,
                                               self.Nt, self.Nx, self.lattice_volume)

    def indices_finder(self, t,x,s, Nx, size_vector = 2):
        return s + x*size_vector + t*Nx*size_vector
    
    def calculate_aggregate(self, size_t, size_x, size_vector, Nt, Nx, lattice_volume):
        size = (size_t*size_x*size_vector)

        nodes_t = np.arange(int(Nt/size_t))*size_t
        nodes_x = np.arange(int(Nx/size_x))*size_x
        row = np.arange(int(lattice_volume*size_vector/size) + 1)*size
        col = []

        Nt_c = int(Nt/size_t) 
        Nx_c = int(Nx/size_x) 
        size_c = Nt_c*Nx_c

        for t in nodes_t:
            for x in nodes_x:
                for i in range(size_x):
                    for j in range(size_t):
                        for s in range(size_vector):
                            col.append(self.indices_finder(t+j, x+i,s, Nx, size_vector))
                
        values = np.ones(len(col))

        return sparse.csr_matrix((values, col, row)).T.tocsr() # , shape = (size_c, lattice_volume*size_vector)

    def start_D_original(self, gauge_links):
        gauge_links_t = gauge_links[:,0,:,:].flatten()
        gauge_links_x = gauge_links[:,1,:,:].flatten()     
        self.calc.UpdateD(gauge_links_t, gauge_links_x)

        self.D_slash_original = self.calc.D_slash
        self.D_slash_dagger_original = self.calc.D_slash_dagger
    
    def update_A_low(self):
        D_factor = scale_factor(self.D_slash_original, max_value= self.max_value)
        D_dagger_factor = scale_factor(self.D_slash_dagger_original, max_value= self.max_value)
        # print(max_complex(np.round(self.D_slash_original*D_factor)))
        D_slash_low = np.round(self.D_slash_original*D_factor)/D_factor + self.diag
        D_slash_dagger_low = np.round(self.D_slash_dagger_original*D_dagger_factor)/D_dagger_factor + self.diag

        self.A_low = D_slash_low.tocsr().dot(D_slash_dagger_low.tocsr())

    def update_A_single(self):
        # print(max_complex(np.round(self.D_slash_original*D_factor)))
        D_slash_low = self.D_slash_original.astype("complex64")+ self.diag
        D_slash_dagger_low = self.D_slash_dagger_original.astype("complex64") + self.diag

        self.A_single = D_slash_low.tocsr().dot(D_slash_dagger_low.tocsr()).astype("complex128")


    def update_m(self,m):
        sparse_identity = sparse.identity(self.D_slash_original.shape[0])
        self.diag = sparse_identity*m
        self.D_slash = self.D_slash_original + self.diag
        self.D_slash_dagger = self.D_slash_dagger_original + self.diag

        A_high = self.D_slash.tocsr().dot(self.D_slash_dagger.tocsr())
        self.A_high = A_high

        self.coefficient = 1.0/approximate_spectral_radius(self.A_high)
        
        self.A_low = self.update_A_low()

        self.A = self.A_high.copy()
        self.update_A_single()