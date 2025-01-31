import numpy as np
from pyamg.util.linalg import norm
from amgqcd.linear_solver.cg import cg


def start_random_guess(self):
    x = np.random.randn(self.A.shape[0], 1).astype(self.A.dtype)
    if self.A.dtype.name.startswith('complex'):
        x = x + 1.0j*np.random.randn(self.A.shape[0], 1).astype(self.A.dtype)
    return x

def initial_setup_stage(self, initial_candidate = None, numPre = 2):
    skip_f_to_i = True

    if initial_candidate is None:
        x = start_random_guess(self)
    else:
        x = np.array(initial_candidate, dtype=self.A.dtype)

    # print(x.dtype)
    for _ in range(numPre):
        self.relax(self.A, x)


    B = x
    self.B = (1.0/norm(B, 'inf')) * B

    self.create_operators(self.B)

def general_setup_iteration(self, numPre = 2, numPost = 2, numCycles = 1, x = None):

    if x is None:
        x = start_random_guess(self)
    x_hat = x.copy()
    B = self.B.copy()

    for i in range(numCycles):

        for _ in range(numPre):
            self.relax(self.A,x)

        residual = -self.A.dot(x)

        coarse_b = self.R.dot(residual)

        coarse_x,info = cg(self.Ac, coarse_b, rtol=self.rtol, maxiter=10000)

        x_correction = self.P.dot(coarse_x)
        x += x_correction


        
        for _ in range(numPost):
            self.relax(self.A,x)



        x = x/norm(x, 'inf')
        self.B = np.hstack((self.B, x.reshape(-1, 1)))
        self.create_operators(self.B)

    if numCycles>1:
        x = self.B[:,-1]
        B = np.hstack((B, x.reshape(-1, 1)))
        self.create_operators(B)


def improve_candidates(self, numPre = 2, numPost = 2, numCycles = 1):
    for i in range(self.B.shape[1]):
        x0 = self.B[:, 0].copy()
        B = np.delete(self.B, 0, axis = -1)
        self.create_operators(B)
        self.general_setup_iteration(numPre=numPre, numPost=numPost, numCycles = numCycles, x = x0)

def generate_candidates(self, numCandidates = 8, numRelax = 100):

    x = start_random_guess(self)

    for i in range(numRelax):
        self.relax(self.A,x)

    x = x/norm(x, 'inf')
    B = x.reshape(-1,1)

    for i in range(numCandidates - 1):
        x = np.random.randn(self.A.shape[0]) + 1j*np.random.randn(self.A.shape[0])
        x_hat = x.copy()
        for i in range(numRelax):
            self.relax(self.A,x)

        x = x/norm(x, 'inf')
        B = np.hstack((B, x.reshape(-1, 1)))
    
    return B

def new_setup(self, initial_relax = True, initial_precision = 6, numCandidates = 8, numRelax = 100,
                second_relax = True, second_precision = 16, numberCycles = 4,
                numPre= 2, numPost= 2):
    self.change_relax_method(initial_relax, initial_precision)

    B = generate_candidates(self, numCandidates = numCandidates, numRelax= numRelax)

    self.change_relax_method(second_relax, second_precision)
    self.counter_mid = 0
    for i in range(len(B[0])):
        x = B[:, 0].copy()
        B = np.delete(B, 0, axis = -1)
        self.create_operators(B)

        for i in range(numberCycles):
            x = self.solve(b = np.zeros_like(x), x = x, numPre= numPre, numPost= numPost)
            self.counter_mid += (numPre + numPost)*2
            self.counter_low -= (numPre + numPost)*2
        
        x = x/norm(x, 'inf')
        B = np.hstack((B, x.reshape(-1, 1)))
        self.create_operators(B)

def new_setup_post_improved(self, initial_relax = True, initial_precision = 6, numCandidates = 8, numRelax = 100,
                second_relax = True, second_precision = 16, numberCycles = 4,
                numPre= 2, numPost= 2):
    self.change_relax_method(initial_relax, initial_precision)

    B = self.generate_candidates(numCandidates = numCandidates, numRelax= numRelax)

    self.create_operators(B)

    self.change_relax_method(second_relax, second_precision)
    self.counter_mid = 0
    B_new = B.copy()
    for i in range(len(B[0])):
        x = B[:, i].copy()

        for _ in range(numberCycles):
            x = self.solve(b = np.zeros_like(x), x = x, numPre=numPre, numPost= numPost)
            self.counter_mid += (numPre + numPost)*2
            self.counter_low -= (numPre + numPost)*2
        
        x = x/norm(x, 'inf')
        B_new[:,i] = x
    
    self.create_operators(B_new)


def smooth_mixed(self,x,b, numIter, numIter2):

    y = np.zeros_like(x)
    for i in range(numIter):
        for j in range(numIter2):
            self.polynomial(self.A,y,b)
        x = x + y
        b = -self.A_mid.dot(x)
        self.counter_mid += 2
        y = np.zeros_like(x)

    
    return x,b

def new_setup_mixed(self, initial_relax = True, initial_precision = 4,
                second_relax = True, second_precision = 16,
                num5 = 1, num10 = 1, num20= 30, numCandidates = 8):
    

    self.change_relax_method(second_relax, second_precision)
    self.A_mid = self.A_low.copy()
    self.counter_mid = 0

    self.change_relax_method(initial_relax, initial_precision)

    x = start_random_guess(self)
    b = -self.A_mid.dot(x)
    self.counter_mid += 2

    x,b = smooth_mixed(self,x,b, num5, 5)
    x,b = smooth_mixed(self,x,b, num10, 10)
    x,b = smooth_mixed(self,x,b, num20, 20)

    x = x/norm(x, 'inf')
    B = x.reshape(-1,1)

    for i in range(numCandidates - 1):
        # print(i)
        x = start_random_guess(self)
        b = -self.A_mid.dot(x)
        self.counter_mid += 2
        x,b = smooth_mixed(self,x,b, num5, 5)
        x,b = smooth_mixed(self,x,b, num10, 10)
        x,b = smooth_mixed(self,x,b, num20, 20)

        x = x/norm(x, 'inf')
        B = np.hstack((B, x.reshape(-1, 1)))

    self.create_operators(B)

class SetupStarter:
    def __init__(self) -> None:
        pass

    def run(self, atg, setup_name = None, initial_relax = None, initial_precision = None, numCandidates = 8, numRelax = None,
                second_relax = None, second_precision = None, numberCycles = None,
                numPre= 2, numPost= 2,
                num5 = 1, num10 = 1, num20= 30):
        
        if setup_name == "mixed" or (setup_name is None):
            if initial_precision is None:
                initial_precision = 4
            if second_precision is None:
                second_precision = 16

            if initial_relax is None:
                initial_relax = True
            if second_relax is None:
                second_relax = True

            new_setup_mixed(atg, initial_relax = initial_relax, initial_precision = initial_precision,
                second_relax = second_relax, second_precision = second_precision,
                num5 = num5, num10 = num10, num20= num20, numCandidates = numCandidates)
            
        elif setup_name == "post_improved":
            if initial_precision is None:
                initial_precision = 6
            if second_precision is None:
                second_precision = 16

            if initial_relax is None:
                initial_relax = True
            if second_relax is None:
                second_relax = True

            if numRelax is None:
                numRelax = 100

            new_setup_post_improved(atg, initial_relax = initial_relax, initial_precision = initial_precision,
                second_relax = second_relax, second_precision = second_precision, 
                numCandidates = numCandidates, numRelax = numRelax,
                numberCycles = 4,
                numPre= numPre, numPost= numPost)
            
        elif setup_name == "standard":
            if initial_precision is None:
                initial_precision = 8
            if initial_relax is None:
                initial_relax = True
            if numRelax is None:
                numRelax = 400

            atg.change_relax_method(initial_relax, initial_precision)
            B = generate_candidates(atg, numCandidates = numCandidates, numRelax = numRelax)
            atg.create_operators(B)
# class SetupStarterMixed(SetupStarterBase):
#     def __init__(self,initial_relax = True, initial_precision = 4,
#                 second_relax = True, second_precision = 16,
#                 num5 = 1, num10 = 1, num20= 15, numCandidates = 8) -> None:
#         super().__init__(numCandidates=numCandidates)

