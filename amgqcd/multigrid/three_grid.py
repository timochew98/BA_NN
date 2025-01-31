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

from .two_grid import TwoGridQCD, TwoGridBase


def Norm(vector):
    vector_flatten = vector.flatten()
    return np.dot(np.conjugate(vector_flatten),vector_flatten)

class ThreeGridQCD(TwoGridQCD):
    def __init__(self, gauge_links, Nt, Nx,Nc,Ns, m = None, size_t = 4, size_x = 4, bit_precision = 5, rtol = 1e-1) -> None:
        super().__init__(gauge_links, Nt, Nx,Nc,Ns, m = m,
                          size_t = size_t, size_x = size_x, bit_precision = bit_precision, rtol = rtol)

    def run_setup_phase(self, setup_name="mixed", **kwargs):
        super().run_setup_phase(setup_name, **kwargs)
        self.tg_second = TwoGridBase(rtol = 1e-1)
        self.tg_second.AggOp = self.calculate_aggregate(self.size_t, self.size_x, self.numCandidates,
                                                         self.Nt//4, self.Nx//4, (self.Nt//4)* (self.Nx//4))
        self.tg_second.update_A(self.Ac)
        self.tg_second.run_setup_phase("standard", initial_precision = None, initial_relax = False, numRelax = 300)

    def solve(self, b, numPre = 2, numPost = 2, print_coarse_iterations = False, x = None, single_precision = True, M = None):
        # return super()._solve(self, b, numPre = numPre, numPost = numPost,
        #                print_coarse_iterations = print_coarse_iterations, psolve = self.psolve_second, x = x,
        #                  single_precision = single_precision)
        self.M = self.tg_second.aspreconditioner(numPre = 1, numPost =1, single_precision=single_precision, print_coarse_iterations=False)
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
                        print_coarse_iterations = print_coarse_iterations, M = self.M, x = x)
        else:
            return self._solve(self.A, self.A_high, self.Ac, self.P, self.R, b,
                        numPre = numPre, numPost = numPost,
                        print_coarse_iterations = print_coarse_iterations, M = self.M, x = x)
    
    # def aspreconditioner(self, numPre = 2, numPost = 2, print_coarse_iterations = False, psolve_preconditioner = None, single_precision = True):
    #     def psolve(b):
    #         print(self.A.shape)
    #         return self.solve(b, numPre = numPre, numPost = numPost,
    #                             print_coarse_iterations = print_coarse_iterations, psolve = psolve_preconditioner, single_precision = single_precision)

    #     return psolve
