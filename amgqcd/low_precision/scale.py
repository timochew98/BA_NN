import numpy as np

def max_abs(matrix):
    return np.max(np.abs(matrix))

def max_complex(matrix):
    return np.max(np.array([max_abs(matrix.real.copy()), max_abs(matrix.imag.copy())]))

def scale_factor(vector, max_value = 2**8-1):
    input_max = max_complex(vector)
    
    if input_max == 0:
        input_max = 1
        
    input_factor = max_value/input_max


    return input_factor