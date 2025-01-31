"""Relaxation methods for linear systems."""

from warnings import warn

import numpy as np
from scipy import sparse
from scipy.linalg import lapack as la

from pyamg.util.utils import type_prep, get_diagonal, get_block_diag
from pyamg.util.params import set_tol
from pyamg.util.linalg import norm, approximate_spectral_radius
from pyamg import amg_core

from amgqcd.low_precision.scale import *

def make_system(A, x, b, formats=None):
    """Return A,x,b suitable for relaxation or raise an exception.

    Parameters
    ----------
    A : sparse-matrix
        n x n system
    x : array
        n-vector, initial guess
    b : array
        n-vector, right-hand side
    formats: {'csr', 'csc', 'bsr', 'lil', 'dok',...}
        desired sparse matrix format
        default is no change to A's format

    Returns
    -------
    (A,x,b), where A is in the desired sparse-matrix format
    and x and b are "raveled", i.e. (n,) vectors.

    Notes
    -----
    Does some rudimentary error checking on the system,
    such as checking for compatible dimensions and checking
    for compatible type, i.e. float or complex.

    """
    if formats is None:
        pass
    elif formats == ['csr']:
        if sparse.isspmatrix_csr(A):
            pass
        elif sparse.isspmatrix_bsr(A):
            A = A.tocsr()
        else:
            warn('implicit conversion to CSR', sparse.SparseEfficiencyWarning)
            A = sparse.csr_matrix(A)
    else:
        if sparse.isspmatrix(A) and A.format in formats:
            pass
        else:
            A = sparse.csr_matrix(A).asformat(formats[0])

    if not isinstance(x, np.ndarray):
        raise ValueError('expected numpy array for argument x')
    if not isinstance(b, np.ndarray):
        raise ValueError('expected numpy array for argument b')

    M, N = A.shape

    if M != N:
        raise ValueError('expected square matrix')

    if x.shape not in [(M,), (M, 1)]:
        raise ValueError('x has invalid dimensions')
    if b.shape not in [(M,), (M, 1)]:
        raise ValueError('b has invalid dimensions')

    # print(A.dtype, x.dtype, b.dtype)
    if A.dtype != x.dtype or A.dtype != b.dtype:
        raise TypeError('arguments A, x, and b must have the same dtype')

    if not x.flags.carray:
        raise ValueError('x must be contiguous in memory')

    x = np.ravel(x)
    b = np.ravel(b)

    return A, x, b



def polynomial(A, x, b, coefficients, iterations=1):
    """Apply a polynomial smoother to the system Ax=b.

    Parameters
    ----------
    A : sparse matrix
        Sparse NxN matrix
    x : ndarray
        Approximate solution (length N)
    b : ndarray
        Right-hand side (length N)
    coefficients : array_like
        Coefficients of the polynomial.  See Notes section for details.
    iterations : int
        Number of iterations to perform

    Returns
    -------
    Nothing, x will be modified in place.

    Notes
    -----
    The smoother has the form  x[:] = x + p(A) (b - A*x) where p(A) is a
    polynomial in A whose scalar coefficients are specified (in descending
    order) by argument 'coefficients'.

    - Richardson iteration p(A) = c_0:
        polynomial_smoother(A, x, b, [c_0])

    - Linear smoother p(A) = c_1*A + c_0:
        polynomial_smoother(A, x, b, [c_1, c_0])

    - Quadratic smoother p(A) = c_2*A^2 + c_1*A + c_0:
        polynomial_smoother(A, x, b, [c_2, c_1, c_0])     # AQUI HAY INFORMACION QUE NECESITO, TIMOTEO

    Here, Horner's Rule is applied to avoid computing A^k directly.

    For efficience, the method detects the case x = 0 one matrix-vector
    product is avoided (since (b - A*x) is b).

    Examples
    --------
    >>> # The polynomial smoother is not currently used directly
    >>> # in PyAMG.  It is only used by the chebyshev smoothing option,
    >>> # which automatically calculates the correct coefficients.
    >>> from pyamg.gallery import poisson
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> from pyamg.aggregation import smoothed_aggregation_solver
    >>> A = poisson((10,10), format='csr')
    >>> b = np.ones((A.shape[0],1))
    >>> sa = smoothed_aggregation_solver(A, B=np.ones((A.shape[0],1)),
    ...         coarse_solver='pinv', max_coarse=50,
    ...         presmoother=('chebyshev', {'degree':3, 'iterations':1}),
    ...         postsmoother=('chebyshev', {'degree':3, 'iterations':1}))
    >>> x0=np.zeros((A.shape[0],1))
    >>> residuals=[]
    >>> x = sa.solve(b, x0=x0, tol=1e-8, residuals=residuals)

    """
    A, x, b = make_system(A, x, b, formats=None)

    # A_factor = scale_factor(A, max_number=2**8 - 1)
    # # print(A_factor)
    # A_low = np.round(A*A_factor)

    for _i in range(iterations):

        if norm(x) == 0:
            residual = b
        else:
            # residual = b - A_low*x/A_factor
            residual = b - A.dot(x)         # TIMOTEO

        h = coefficients[0]*residual

        for c in coefficients[1:]:
            # h = c*residual + A_low*h/A_factor
            h = c*residual + A.dot(h)

        x += h.squeeze()
    
    # print("high")
    # print(norm(A.dot(x)))

def polynomial_low_matrix(A, x, b, coefficients, iterations=1, bit_precision = 5):

    A, x, b = make_system(A, x, b, formats=None)

    A_factor = scale_factor(A, max_number=2**(bit_precision-1) - 1)
    # print(A_factor)
    A_low = np.round(A*A_factor)

    for _i in range(iterations):

        if norm(x) == 0:
            residual = b
        else:
            
            residual = b - A_low*x/A_factor
            # residual = b - A*x         # TIMOTEO

        h = coefficients[0]*residual

        for c in coefficients[1:]:
            h = c*residual + A_low*h/A_factor
            # h = c*residual + A*h

        x += h

    print("IMPOSTOR")
    # print(norm(A.dot(x)))


# counter = 0 
def polynomial_low_both(A, x, b, coefficients, iterations=1, bit_precision = 6, noise_strength = None):
    max_value = 2**(bit_precision-1) - 1

    A, x, b = make_system(A, x, b, formats=None)

    A_factor = scale_factor(A, max_value=max_value)
    # print(A_factor)
    A_low = np.round(A*A_factor)
    # print(max_complex(A_low))
    if noise_strength is not None:
        A_low.data += np.random.randint(low = -noise_strength, high = noise_strength + 1, size = A_low.data.shape)

    # print("Low")


    for _i in range(iterations):

        if norm(x) == 0:
            residual = b
        else:
            x_factor = scale_factor(x, max_value=max_value)
            x_low = np.round(x*x_factor)
            # print(max_complex(x_low))
            if noise_strength is not None:
                x_low += np.random.randint(low = -noise_strength, high = noise_strength + 1, size = x_low.shape)

            # print(np.max(np.abs(x_low)))
            # print(np.max(np.abs(A_low)))

            residual = b - A_low.dot(x_low)/(A_factor*x_factor)
            # residual = b - A*x         # TIMOTEO

        h = coefficients[0]*residual

        for c in coefficients[1:]:
            h_factor = scale_factor(h, max_value=max_value)
            h_low = np.round(h*h_factor)
            if noise_strength is not None:
                h_low += np.random.randint(low = -noise_strength, high = noise_strength + 1, size = h_low.shape)

            h = c*residual + A_low.dot(h_low)/(A_factor*h_factor)
            # h = c*residual + A*h

        x += h

    # counter += 1
    # print(counter)
    # print("Low")
    
    # print(norm(A.dot(x)))

def polynomial_low_vector(A, x, b, coefficients, iterations=1, bit_precision = 5, noise_strength = None):

    A, x, b = make_system(A, x, b, formats=None)
    # if noise_strength is not None:
    #     A_low.data += np.random.randint(low = -noise_strength, high = noise_strength + 1, size = A_low.data.shape)

    # print("Low")


    for _i in range(iterations):

        if norm(x) == 0:
            residual = b
        else:
            x_factor = scale_factor(x, max_value=2**(bit_precision-1) - 1)
            x_low = np.round(x*x_factor)
            # print(max_complex(x_low))
            # print(np.max(np.abs(x_low)))
            # print(np.max(np.abs(A_low)))

            residual = b - A.dot(x_low)/(x_factor)
            # residual = b - A*x         # TIMOTEO

        h = coefficients[0]*residual

        for c in coefficients[1:]:
            h_factor = scale_factor(h, max_value=2**(bit_precision-1) - 1)
            h_low = np.round(h*h_factor)
            h = c*residual + A.dot(h_low)/(h_factor)
            # h = c*residual + A*h

        x += h



def richardson_prolongation_smoother(S, T, omega=4.0/3.0, degree=1):
    weight = omega/approximate_spectral_radius(S)

    P = T
    for _ in range(degree):
        P = P - weight*(S.dot(P))

    return P