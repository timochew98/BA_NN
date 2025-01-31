import numpy as np

def Norm(vector):
    return np.dot(np.conjugate(vector),vector)

def SP(vector1, vector2):
    return np.dot(np.conjugate(vector1),vector2)

def SquareDistance(array1,array2, rounding = 18):
    return np.sqrt(np.sum((np.round((array1-array2)*(10**rounding)))**2))/(10**rounding)

def Transpose(array):
    axes = np.arange(len(array.shape))
    axes[-2:] = np.flip(axes[-2:]) 
    return np.transpose(array, axes = axes)

def ConjugateTranspose2(array):
    axes = np.arange(len(array.shape))
    axes[-2:] = np.flip(axes[-2:]) 
    return np.conjugate(np.transpose(array, axes = axes))