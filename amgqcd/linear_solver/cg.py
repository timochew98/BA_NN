from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg._isolve.utils import make_system as make_system_cg

import numpy as np

from amgqcd.low_precision.scale import *


def get_atol_rtol(name, b_norm, atol=0., rtol=1e-5):
    """
    A helper function to handle tolerance normalization
    """
    if atol == 'legacy' or atol is None or atol < 0:
        msg = (f"'scipy.sparse.linalg.{name}' called with invalid `atol`={atol}; "
               "if set, `atol` must be a real, non-negative number.")
        raise ValueError(msg)

    atol = max(float(atol), float(rtol) * float(b_norm))

    return atol, rtol

from timeit import default_timer as timer


# def time_function(my_func, *args):
#     start = timer()
#     result = my_func(*args)
#     end = timer()
#     print(end-start)
#     return result

# def cg(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None, callback=None):

#     A, M, x, b, postprocess = time_function(make_system_cg, A, M, x0, b)
#     bnrm2 = np.linalg.norm(b)

#     atol, _ = get_atol_rtol('cg', bnrm2, atol, rtol)

#     if bnrm2 == 0:
#         return postprocess(b), 0

#     n = len(b)

#     if maxiter is None:
#         maxiter = n*10

#     dotprod = np.vdot if np.iscomplexobj(x) else np.dot

#     matvec = A.matvec
#     psolve = M.matvec
#     r = b - time_function(matvec, x) if x.any() else b.copy()

#     # Dummy value to initialize var, silences warnings
#     rho_prev, p = None, None

#     for iteration in range(maxiter):
#         # print(iteration)
#         if np.linalg.norm(r) < atol:  # Are we done?
#             return postprocess(x), iteration
#         print("psolve")
#         z = time_function(psolve, r)
#         print("dotprod")
#         rho_cur = time_function(dotprod,r, z)
#         if iteration > 0:
#             beta = rho_cur / rho_prev
#             p *= beta
#             p += z
#         else:  # First spin
#             p = np.empty_like(r)
#             p[:] = z[:]
#         print("matvec")
#         q = time_function(matvec, p)
#         alpha = rho_cur / dotprod(p, q)
#         x += alpha*p
#         r -= alpha*q
#         rho_prev = rho_cur

#         if callback:
#             callback(x)

#     else:  # for loop exhausted
#         # Return incomplete progress
#         return postprocess(x), maxiter

def cg(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None, callback=None):

    A, M, x, b, postprocess = make_system_cg(A, M, x0, b)
    bnrm2 = np.linalg.norm(b)

    atol, _ = get_atol_rtol('cg', bnrm2, atol, rtol)

    if bnrm2 == 0:
        return postprocess(b), 0

    n = len(b)

    if maxiter is None:
        maxiter = n*10

    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    matvec = A.matvec
    psolve = M.matvec
    r = b - matvec(x) if x.any() else b.copy()

    # Dummy value to initialize var, silences warnings
    rho_prev, p = None, None

    for iteration in range(maxiter):
        # print(iteration)
        if np.linalg.norm(r) < atol:  # Are we done?
            return postprocess(x), iteration

        z = psolve(r)
        rho_cur = dotprod(r, z)
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:  # First spin
            p = np.empty_like(r)
            p[:] = z[:]

        q = matvec(p)
        alpha = rho_cur / dotprod(p, q)
        x += alpha*p
        r -= alpha*q
        rho_prev = rho_cur

        if callback:
            callback(x)

    else:  # for loop exhausted
        # Return incomplete progress
        return postprocess(x), maxiter
    
# def cg(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, psolve=None, callback=None):
#     if psolve is None:
#         psolve = lambda b : b

#     if x0 is None:
#         x = np.zeros_like(b)
#     else:
#         x = x0.copy()
#     bnrm2 = np.linalg.norm(b)

#     atol, _ = get_atol_rtol('cg', bnrm2, atol, rtol)

#     if bnrm2 == 0:
#         return b, 0

#     n = len(b)

#     if maxiter is None:
#         maxiter = n*10

#     dotprod = np.vdot if np.iscomplexobj(x) else np.dot

#     matvec = A.dot
#     r = b - matvec(x) if x.any() else b.copy()

#     # Dummy value to initialize var, silences warnings
#     rho_prev, p = None, None

#     for iteration in range(maxiter):
#         # print(iteration)
#         if np.linalg.norm(r) < atol:  # Are we done?
#             return x, iteration

#         z = psolve(r)
#         rho_cur = dotprod(r, z)
#         if iteration > 0:
#             beta = rho_cur / rho_prev
#             p *= beta
#             p += z
#         else:  # First spin
#             p = np.empty_like(r)
#             p[:] = z[:]

#         q = matvec(p)
#         alpha = rho_cur / dotprod(p, q)
#         x += alpha*p
#         r -= alpha*q
#         rho_prev = rho_cur

#         if callback:
#             callback(x)

#     else:  # for loop exhausted
#         # Return incomplete progress
#         return x, maxiter
    

def lcg(A, b, bit_precision = 5, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None, callback=None):

    max_value = 2**(bit_precision-1) - 1

    A_factor = scale_factor(A, max_value=max_value)
    A_low = np.round(A*A_factor)/A_factor

    A_low, M, x, b, postprocess = make_system_cg(A_low, M, x0, b)
    bnrm2 = np.linalg.norm(b)

    atol, _ = get_atol_rtol('cg', bnrm2, atol, rtol)

    if bnrm2 == 0:
        return postprocess(b), 0

    n = len(b)

    if maxiter is None:
        maxiter = n*10

    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    matvec = A_low.matvec
    psolve = M.matvec
    r = b - matvec(x) if x.any() else b.copy()

    # Dummy value to initialize var, silences warnings
    rho_prev, p = None, None

    for iteration in range(maxiter):
        if np.linalg.norm(r) < atol:  # Are we done?
            return postprocess(x), iteration

        z = psolve(r)
        rho_cur = dotprod(r, z)
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:  # First spin
            p = np.empty_like(r)
            p[:] = z[:]

        p_factor = scale_factor(p, max_value=max_value)
        p_low = np.round(p*p_factor)/p_factor
        q = matvec(p_low)
        alpha = rho_cur / dotprod(p, q)
        x += alpha*p
        r -= alpha*q
        rho_prev = rho_cur

        if callback:
            callback(x)

    else:  # for loop exhausted
        # Return incomplete progress
        return postprocess(x), maxiter


def lcg2(A, b, bit_precision = 5, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None, callback=None):

    max_value = 2**(bit_precision-1) - 1

    A_factor = scale_factor(A, max_value=max_value)
    A_low = np.round(A*A_factor)/A_factor

    A_low, M, x, b, postprocess = make_system_cg(A_low, M, x0, b)
    bnrm2 = np.linalg.norm(b)

    atol, _ = get_atol_rtol('cg', bnrm2, atol, rtol)

    if bnrm2 == 0:
        return postprocess(b), 0

    n = len(b)

    if maxiter is None:
        maxiter = n*10

    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    matvec = A_low.matvec
    psolve = M.matvec
    r = b - matvec(x) if x.any() else b.copy()

    # Dummy value to initialize var, silences warnings
    rho_prev, p = None, None

    for iteration in range(maxiter):
        if np.linalg.norm(r) < atol:  # Are we done?
            return postprocess(x), iteration

        
        z = psolve(r)
        rho_cur = dotprod(r, z)
        if iteration > 0:
            beta = (rho_cur-rho_cur_old) / rho_prev
            p *= beta
            p += z
        else:  # First spin
            p = np.empty_like(r)
            p[:] = z[:]

        p_factor = scale_factor(p, max_value=max_value)
        p_low = np.round(p*p_factor)/p_factor
        q = matvec(p_low)
        alpha = rho_cur / dotprod(p, q)
        x += alpha*p
        r -= alpha*q
        rho_prev = rho_cur

        rho_cur_old = dotprod(r, z)

        if callback:
            callback(x)

    else:  # for loop exhausted
        # Return incomplete progress
        return postprocess(x), maxiter