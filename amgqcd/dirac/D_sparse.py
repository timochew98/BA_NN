from scipy import sparse
import numpy as np
from scipy.stats import unitary_group

from amgqcd.linalg.dirac_functions import *

# Pauli = []
# Pauli.append(np.array([[0, 1], [1, 0]]))
# Pauli.append(np.array([[0,-1j], [1j, 0]]))
# Pauli.append(np.array([[1, 0], [0, -1]]))
# Id = np.array([[1, 0], [0, 1]])

def roll_x_indices(Nt,Nx,size_matrix,roll):
    size0 = Nx*size_matrix

    indices_t = np.arange(Nt) *size0

    indices_x = np.arange(Nx)*size_matrix
    rolled_indices_x = np.roll(indices_x,roll)

    indices = np.arange(size_matrix)

    return ((indices_t[:,None] + rolled_indices_x[None,:])[:,:,None] + indices[None,:]).flatten()


def roll_t_indices(Nt,Nx,size_matrix,roll):
    size0 = Nx*size_matrix

    indices_t = np.arange(Nt) *size0
    rolled_indices_t = np.roll(indices_t,roll)

    indices_x = np.arange(Nx)*size_matrix


    indices = np.arange(size_matrix)

    roll_indices = ((rolled_indices_t[:,None] + indices_x[None,:])[:,:,None] + indices[None,:])

    return roll_indices.flatten(), roll_indices[int(roll*0.5 - 0.5)].flatten()

class IndicesCreator:
    def MatMul_gamma5_indices(self,Nt,Nx,Nc,Ns):
        gamma5_indices = np.repeat(np.repeat((np.arange(Ns)*Ns).reshape(1,Ns),Nc,0).reshape(1,Ns*Nc),Nt*Nx,0).flatten()

        size_matrix = Nc*Ns
        indices2 = np.repeat((np.arange(Nc)*Ns),Ns).flatten()
        indices_lattice = np.arange(Nt*Nx)*size_matrix
        psi_indices = (indices_lattice[:,None] +indices2[None, :]).flatten()

        return gamma5_indices, psi_indices


    def roll_x_indices(self,Nt,Nx,size_matrix,roll):
        size0 = Nx*size_matrix

        indices_t = np.arange(Nt) *size0

        indices_x = np.arange(Nx)*size_matrix
        rolled_indices_x = np.roll(indices_x,roll)

        indices = np.arange(size_matrix)

        return ((indices_t[:,None] + rolled_indices_x[None,:])[:,:,None] + indices[None,:]).flatten()


    def roll_t_indices(self,Nt,Nx,size_matrix,roll):
        size0 = Nx*size_matrix

        indices_t = np.arange(Nt) *size0
        rolled_indices_t = np.roll(indices_t,roll)

        indices_x = np.arange(Nx)*size_matrix


        indices = np.arange(size_matrix)

        roll_indices = ((rolled_indices_t[:,None] + indices_x[None,:])[:,:,None] + indices[None,:])

        return roll_indices.flatten(), roll_indices[int(roll*0.5 - 0.5)].flatten()

    def Transposed_indices(self,Nt,Nx,nDim):
        indices = np.arange(nDim)
        repeated_indices = indices*nDim
        size_matrix = nDim**2


        indices_lattice = np.arange(Nt*Nx)*size_matrix
        #indices_lattice = np.arange(Nt*Nx) * size0
        #print(indices_lattice)
        #print((indices_t[:,None] + indices_x[None,:]))
        return ((indices_lattice[:,None] + indices[None,:])[:,:,None] + repeated_indices[None,:]).flatten()

    def MatMul_indices(self,Nt,Nx,nDim):
        indices = np.arange(nDim)
        #repeated_indices = indices*nDim
        columns = indices*nDim
        size_matrix = nDim**2

        #print(repeated_indices)
        #columns = indices[:,None] + repeated_indices[None,:]
        indices_lattice = np.arange(Nt*Nx)*size_matrix
        #indices_lattice = np.arange(Nt*Nx) * size0
        #print(indices_lattice)
        #print((indices_t[:,None] + indices_x[None,:]))
    #     matmul_indices = np.zeros((nDim, Nt*Nx*nDim), dtype = 'int')
        indices_column = (indices_lattice[:,None] +columns[None, :]).flatten()
        size_columns = Nt*Nx*nDim
    #     for numColumn in range(nDim):
    #         indices_column = ((indices_t[:,None] + indices_x[None,:])[:,:,None] +columns[None,numColumn, :]).flatten()
    #         matmul_indices[numColumn] = indices_column

        #Indices for matrix2
        columns2 = np.arange(Nt*Nx)*nDim
        columns_matrix2 = np.repeat(columns2, nDim)
        return indices_column, columns_matrix2

    def Kron_indices(self,Nt,Nx, Nc,Ns):
        lattice_volume = Nt*Nx
        gauge_links_indices = np.reshape(np.arange(lattice_volume*Nc*Nc),(lattice_volume,Nc,Nc))
        repeated_gauge_links_indices =  np.repeat(np.repeat(gauge_links_indices,Ns,2),Ns,1)

        gamma_indices = np.reshape(np.arange(Ns*Ns), (Ns,Ns))
        repeated_gamma_indices = np.repeat(np.repeat(gamma_indices,Nc,0).reshape(1,Ns*Ns*Nc),lattice_volume*Nc,0)
        return repeated_gauge_links_indices.flatten() , repeated_gamma_indices.flatten()

    def indices_flatten(self,indices,size_matrix): # Add self

        indices_helper = np.arange(size_matrix)

        indices_unflatten = ((indices*size_matrix)[:,None] + indices_helper[None,:])

        return indices_unflatten.flatten()
    

class CalculatorD(IndicesCreator):
    def __init__(self, Nt,Nx,Nc,Ns, m):
        self.Nt = Nt
        self.Nx = Nx
        self.Nc = Nc
        self.Ns = Ns
        self.m = m
        self.lattice_volume = Nt*Nx
        
        self.Pauli = []
        self.Pauli.append(np.array([[0, 1], [1, 0]]))
        self.Pauli.append(np.array([[0,-1j], [1j, 0]]))
        self.Pauli.append(np.array([[1, 0], [0, -1]]))
        self.Id = np.array([[1, 0], [0, 1]])

        self.gamma5 = np.kron(np.identity(Nc), 1j*np.matmul(self.Pauli[2], self.Pauli[0]))
        self.Gamma5 = self.CalculateGamma5()

        self.down_t_gauge, self.down_t_gauge_antiperiodic = self.roll_t_indices(Nt,Nx,Nc*Nc,1)
        self.down_x_gauge                    = self.roll_x_indices(Nt,Nx,Nc*Nc,1)
        self.up_t_gauge, self.up_t_gauge_antiperiodic     = self.roll_t_indices(Nt,Nx,Nc*Nc,-1)
        self.up_x_gauge                      = self.roll_x_indices(Nt,Nx,Nc*Nc,-1)

        self.up_t_psi, self.up_t_psi_antiperiodic         = self.roll_t_indices(Nt,Nx,Nc*Ns,-1)
        self.up_x_psi                        = self.roll_x_indices(Nt,Nx,Nc*Ns,-1)
        self.down_t_psi, self.down_t_psi_antiperiodic     = self.roll_t_indices(Nt,Nx,Nc*Ns,1)
        self.down_x_psi                      = self.roll_x_indices(Nt,Nx,Nc*Ns,1)

        self.transposed_indices = self.Transposed_indices(Nt,Nx,Nc)

        self.matmul_indices, self.columns_matrix2 = self.MatMul_indices(Nt,Nx,Nc*Ns)

        self.gauge_links_indices, self.gamma_indices = self.Kron_indices(Nt,Nx, Nc,Ns)

        

        self.gamma_t_plus = np.identity(Ns) - self.Pauli[2] 
        self.gamma_t_minus = np.identity(Ns) + self.Pauli[2] 
        self.gamma_x_plus = np.identity(Ns) - self.Pauli[0] 
        self.gamma_x_minus = np.identity(Ns) + self.Pauli[0]

        self.gamma5_indices, self.psi_indices = self.MatMul_gamma5_indices(Nt,Nx,Nc,Ns)
        self.row_index, self.col_index = self.CalculateD_indices()

        self.D_slash = None
        self.values_indices = self.start_values_indices()
    def start_values_indices(self):

        nDim = 2
        numCol = (2**nDim) # Without diagonal

        indices1 = np.arange(self.lattice_volume * (self.Nc*self.Ns))*((self.Nc*self.Ns)) #Explore the lattice points inside of one of the nn, indices of all?
        indices2 = np.arange(numCol)*self.lattice_volume*(self.Nc*self.Ns)**2 # the indices of the starting point of each of the nn
        indices3 = np.arange((self.Nc*self.Ns)) # Indices of the one row of the mini matrices when being in one lattice point
        indices = indices3[None,:] + indices2[:,None]
        indices = indices.flatten()[None,:] + indices1[:,None]

        return indices
    
    def CalculateDiag(self, m):
        #In each row they are 2^d (next Neighbors) + 1 (onsite)
        row_index = np.arange(0,(1)*self.lattice_volume+1, 1)
        

        col_index = np.arange(0,(1)*self.lattice_volume, 1)
        value = np.identity(self.Nc*self.Ns)*(m+2)
        values = []
        for i in range(self.lattice_volume):
            values.append(value)
        
        return sparse.bsr_array(((values, col_index, row_index)), shape = (self.lattice_volume*self.Nc*self.Ns, self.lattice_volume*self.Nc*self.Ns))
        
    def CalculateGamma5(self):
        #In each row they are 2^d (next Neighbors) + 1 (onsite)
        row_index = np.arange(0,(1)*self.lattice_volume+1, 1)
        

        col_index = np.arange(0,(1)*self.lattice_volume, 1)

        values = []
        for i in range(self.lattice_volume):
            values.append(self.gamma5)
        
        return sparse.bsr_array(((values, col_index, row_index)), shape = (self.lattice_volume*self.Nc*self.Ns, self.lattice_volume*self.Nc*self.Ns))

    def Roll(self,matrix, indices):
        return matrix[indices]

    def ConjugateTranspose(self,matrix,transposed_indices): 
        return np.conjugate(matrix[transposed_indices])

    def MatMul(self,matrix1, matrix2, matmul_indices, columns_matrix2, nDim):
        result = np.zeros(len(matmul_indices), dtype = 'complex_')
        for column in range(nDim):
            result = result + (matrix1[matmul_indices +column]*matrix2[columns_matrix2 + column])           
        return result

    def Kron(self,matrix1,matrix2,indices1,indices2):
        return matrix1[indices1]*matrix2[indices2]
    
    def Antiperiodic(self, matrix, indices):
        matrix_result = np.copy(matrix)
        matrix_result[indices] = -matrix_result[indices]
        
        return matrix_result
    
    def prepareD(self,gauge_links_t, gauge_links_x):
        #Symetric derivative, each dimension multiplied by a gamma matrix, in 2D, and in this choice, they are pauli matrices

        #Correct this
        offdiagonal_spinor_x_plus = -0.5*(self.Id - self.Pauli[0]).flatten()
        offdiagonal_spinor_x_minus = -0.5*(self.Id + self.Pauli[0]).flatten()
        offdiagonal_spinor_t_plus = -0.5*(self.Id - self.Pauli[2]).flatten()
        offdiagonal_spinor_t_minus = -0.5*(self.Id + self.Pauli[2]).flatten()

        gauge_links_shifted_t = self.Roll(gauge_links_t.copy(),self.down_t_gauge)
        gauge_links_shifted_x = self.Roll(gauge_links_x.copy(),self.down_x_gauge)

        # gauge_links_shifted_t[:self.Nx] =  -gauge_links_shifted_t[:self.Nx]
        # gauge_links_shifted_x[::self.Nt] =  -gauge_links_shifted_x[::self.Nt]

        offdiag_x_plus = self.Kron(gauge_links_x,offdiagonal_spinor_x_plus, self.gauge_links_indices, self.gamma_indices).reshape(self.Nt*self.Nx,
                                                                                                                                  self.Nc*self.Ns,
                                                                                                                                  self.Nc*self.Ns) #Careful with U dimension t,x
        offdiag_x_minus = self.Kron(self.ConjugateTranspose(gauge_links_shifted_x,self.transposed_indices),offdiagonal_spinor_x_minus,
                                     self.gauge_links_indices, self.gamma_indices).reshape(self.Nt*self.Nx,
                                                                                        self.Nc*self.Ns,
                                                                                        self.Nc*self.Ns)

        offdiag_t_plus = self.Kron(gauge_links_t, offdiagonal_spinor_t_plus, self.gauge_links_indices, self.gamma_indices).reshape(self.Nt*self.Nx,
                                                                                                                                  self.Nc*self.Ns,
                                                                                                                                  self.Nc*self.Ns)
        offdiag_t_minus = self.Kron(self.ConjugateTranspose(gauge_links_shifted_t, self.transposed_indices),offdiagonal_spinor_t_minus, self.gauge_links_indices,
                                     self.gamma_indices).reshape(self.Nt*self.Nx,
                                                                self.Nc*self.Ns,
                                                                self.Nc*self.Ns)

        return offdiag_x_plus, offdiag_x_minus, offdiag_t_plus, offdiag_t_minus
    
    def CalculateD(self, gauge_links_t, gauge_links_x):
        #Delta_x_y, dependent only on the onsite


        nDim = 2
        numCol = (2**nDim)
        
        self.offdiag_x_plus, self.offdiag_x_minus, self.offdiag_t_plus, self.offdiag_t_minus = self.prepareD(gauge_links_t, gauge_links_x)

        values = np.append(self.offdiag_t_minus, (self.offdiag_x_minus, self.offdiag_x_plus, self.offdiag_t_plus))
        values = values[self.values_indices.flatten()]

        # values = np.reshape(values, (self.lattice_volume*numCol, self.Nc*self.Ns, self.Nc*self.Ns))
        
        antiperiodic_m_t = np.arange(self.Nx*self.Nc*self.Ns)*numCol*self.Nc*self.Ns
        indices_helper = np.arange(self.Nc*self.Ns)
        antiperiodic_m_t = (antiperiodic_m_t[:,np.newaxis] + indices_helper).flatten()

        indices_helper = np.arange(self.Nc*self.Ns)*(-1)
        antiperiodic_p_t = (np.arange(self.Nx*self.Nc*self.Ns)*-numCol*self.Nc*self.Ns)-1
        antiperiodic_p_t = (antiperiodic_p_t[:,np.newaxis] + indices_helper).flatten()
        
        values[antiperiodic_m_t] = -values[antiperiodic_m_t]
        values[antiperiodic_p_t] = -values[antiperiodic_p_t]
        
        # values_ = values.copy()
        values = values.flatten()
        # print(values.shape)

        D = sparse.csr_array(((values, self.col_index, self.row_index)), shape = (self.lattice_volume*self.Nc*self.Ns, self.lattice_volume*self.Nc*self.Ns))
        D += self.CalculateDiag(self.m)
        return D #, values , values_
    
    def UpdateD(self, gauge_links_t, gauge_links_x):
        self.D_slash = self.CalculateD(gauge_links_t, gauge_links_x)
        self.D_slash_dagger = self.Gamma5.dot(self.D_slash.dot(self.Gamma5))

    def applyD(self,psi):
        return self.D_slash.dot(psi)
    
    # def applyD_dagger(self,psi):
    #     result = self.Gamma5.dot(psi)
    #     result = self.D_slash.dot(result)
    #     return self.Gamma5.dot(result)

    def applyD_dagger(self,psi):
        return self.D_slash_dagger.dot(psi)
    
    # def start_hermitian_matrix(self, gauge_links_t, gauge_links_x):
    #     self.UpdateD(gauge_links_t, gauge_links_x)
    #     array = self.D_slash
    #     gamma5 = self.Gamma5
    #     return array.dot(gamma5.dot(array.dot(gamma5)))
    

    def CalculateD_indices_old(self):
        #Delta_x_y, dependent only on the onsite
        
        nDim = 2
        #In each row they are 2^d (next Neighbors) + 1 (onsite)
        row_index = np.arange(0,(2**nDim)*self.lattice_volume+1, 2**nDim)
        
        roll_x_m = roll_x_indices(self.Nt,self.Nx,1,1)
        roll_x_p = roll_x_indices(self.Nt,self.Nx,1,-1)
        roll_t_m, _ = roll_t_indices(self.Nt,self.Nx,1,1)
        roll_t_p, _ = roll_t_indices(self.Nt,self.Nx,1,-1)

        col_index = np.stack((roll_t_m, roll_x_m, roll_x_p, roll_t_p)).T.flatten()
        
        return row_index.astype(int), col_index.astype(int)

    def CalculateD_indices_new(self):
        #Delta_x_y, dependent only on the onsite
        
        nDim = 2
        #In each row they are 2^d (next Neighbors) + 1 (onsite)
        row_index = np.arange(0,(2**nDim)*self.lattice_volume+1, 2**nDim)
        
        roll_x_m = roll_x_indices(self.Nt,self.Nx,1,1)
        roll_x_p = roll_x_indices(self.Nt,self.Nx,1,-1)
        roll_t_m, _ = roll_t_indices(self.Nt,self.Nx,1,1)
        roll_t_p, _ = roll_t_indices(self.Nt,self.Nx,1,-1)

        col_index = np.zeros(self.lattice_volume*(2**nDim), dtype = 'int')
        col_index[0::4] = roll_t_m
        col_index[1::4] = roll_x_m
        col_index[2::4] = roll_x_p
        col_index[3::4] = roll_t_p

        return row_index.astype(int), col_index.astype(int)

    def CalculateD_indices(self):
        #Delta_x_y, dependent only on the onsite
        
        nDim = 2
        #In each row they are 2^d (next Neighbors) + 1 (onsite)
        row_index = np.arange((self.lattice_volume)*(self.Nc*self.Ns) + 1)*((self.Nc*self.Ns)*2**nDim)
        
        # roll_x_m = roll_x_indices(self.Nt,self.Nx,1,1)
        # roll_x_p = roll_x_indices(self.Nt,self.Nx,1,-1)
        # roll_t_m, _ = roll_t_indices(self.Nt,self.Nx,1,1)
        # roll_t_p, _ = roll_t_indices(self.Nt,self.Nx,1,-1)

        # roll_x_m = self.indices_flatten(roll_x_m, self.Nc*self.Ns)
        # roll_x_p = self.indices_flatten(roll_x_p, self.Nc*self.Ns)
        # roll_t_m = self.indices_flatten(roll_t_m, self.Nc*self.Ns)
        # roll_t_p = self.indices_flatten(roll_t_p, self.Nc*self.Ns)

        _, col_old = self.CalculateD_indices_old()
        col_old = self.indices_flatten(col_old, self.Nc*self.Ns).reshape(self.lattice_volume, self.Nc*self.Ns*(2**nDim))
        col_index = np.repeat(col_old, self.Nc*self.Ns, axis = 0).flatten()

        # colPerRow = (2**nDim)*(self.Nc*self.Ns)
        # numRows = self.lattice_volume*(self.Nc*self.Ns)
        # col_index = np.zeros((numRows)*(colPerRow), dtype = 'int').reshape(colPerRow, numRows)
        # sizevector = (self.Nc*self.Ns)
        # for i in range(sizevector):
        #     col_index[0 + i, :] = roll_t_m
        #     col_index[sizevector*1 + i, :] = roll_x_m
        #     col_index[sizevector*2 + i, :] = roll_x_p
        #     col_index[sizevector*3 + i, :] = roll_t_p
        # col_index = col_index.T.flatten()
        
        return row_index.astype(int), col_index.astype(int)
    
    def applyD_inverse(self,psi,x0 = None, targetResidue = 1e-15, iterMax = 5000):
        if x0 is None:
            x0 = np.zeros_like(psi)
        result_guess = self.applyD(self.applyD_dagger(x0))
        r = psi - result_guess
        p = r
        x = x0
        
        counter = 0
        beta_nominator = 1e10
        while (beta_nominator > targetResidue and counter < iterMax):
            #tmp
            tmp = self.applyD_dagger(p)
            
        

            #Alpha
            alpha_denominator = Norm(tmp)
            alpha_nominator = Norm(r)
            alpha = alpha_nominator/alpha_denominator

            #Updates
            x = x + alpha* p
            
            residue_update = self.applyD(tmp)
            
#             print(tmp)
#             print(residue_update)
#             hola
            
            
            r = r - alpha * residue_update

            #Beta
            beta_nominator = Norm(r)
            beta = beta_nominator/alpha_nominator

            p = r + beta*p
            counter = counter + 1
            # if counter%(iterMax/10) == 0:
            #     print('Iteration # {}'.format(counter))
        print('Number of iteractions: {}'.format(counter))
        return x, beta_nominator