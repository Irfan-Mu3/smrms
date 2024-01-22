import random as rd
import time
import numpy as np
from numba import jit, njit
from scipy.signal import deconvolve

import pyfftw as fftw

fftw.interfaces.cache.enable()
pfftw = fftw.interfaces.numpy_fft.fft
ipfftw = fftw.interfaces.numpy_fft.ifft

"""
Solves CONTINUOUS smrm reachability problems: The probability of reaching a set of states B, with reward cumulated between
the range [0, b].

To see how an sMRM is created, please see smrm_usecase_examples.py. The simplest case is to use the power method
 with the following data: A ,b , S_matrix and s_vector, N, xs. Solvers involve two functions: preparation and solving.

Note that preparations that do not contain the 'no_pdf' suffix requires the S_matrix to be a matrix of objects each 
with a callable pdf() method. For preparation methods without this suffix, then S_matrix represents a hypermatrix 
where each object is replaced with a vector of sampled points, each of same length. 

Unfortunately, at this time, almost all algorithms require that each state can reach an absorbing state. 
The only exception is 'continuous_prepare_AoC_no_pdf_w_s1' for the continuous case, with the trapezoid/Romberg power method. 
In the continuous case, when using the trapezoid rule, two preparations are required.
"""


####################################################################################################
# Preparation algorithms

def continuous_prepare_AoC(A_matrix, b_vector, S_matrix, s_vector, N, xs, scaled_weights=None, shifted=False):
    num_states = len(A_matrix[0])
    padded_N = 2 * N - 1
    zero_pads = np.zeros(N - 1)

    C_matrix = np.empty((padded_N, num_states, num_states), dtype=complex)  # FFT matrix
    d_vector = np.empty((padded_N, num_states, 1), dtype=complex)  # FFT vector

    for i in range(num_states):
        d_vector[:, i, 0] = pfftw(np.append(b_vector[i] * s_vector[i].pdf(xs), zero_pads))
        for j in range(num_states):
            if scaled_weights is not None:
                if shifted:
                    scaled_weights[0:1] = 0
                C_matrix[:, i, j] = pfftw(np.append(S_matrix[i, j].pdf(xs) * scaled_weights, zero_pads))
            else:
                C_matrix[:, i, j] = pfftw(np.append(S_matrix[i, j].pdf(xs), zero_pads))

    AoC = A_matrix * C_matrix

    return AoC, d_vector


def continuous_prepare_AoC_no_pdfs(A_matrix, b_vector, S_matrix, s_vector, N, xs, scaled_weights=None, shifted=False):
    num_states = len(A_matrix[0])
    padded_N = 2 * N - 1
    zero_pads = np.zeros(N - 1)

    C_matrix = np.empty((padded_N, num_states, num_states), dtype=complex)  # FFT matrix
    d_vector = np.empty((padded_N, num_states, 1), dtype=complex)  # FFT vector

    for i in range(num_states):
        d_vector[:, i, 0] = pfftw(np.append(b_vector[i] * s_vector[i](xs), zero_pads))
        for j in range(num_states):
            if scaled_weights is not None:
                if shifted:
                    scaled_weights[0:1] = 0
                C_matrix[:, i, j] = pfftw(np.append(S_matrix[i, j](xs) * scaled_weights, zero_pads))
            else:
                C_matrix[:, i, j] = pfftw(np.append(S_matrix[i, j](xs), zero_pads))

    AoC = A_matrix * C_matrix

    return AoC, d_vector


def continuous_prepare_AoC_no_pdf_w_s1(A_matrix, s1b_vector, S_matrix, s_vector, N, xs, scaled_weights=None,
                                       shifted=False):
    num_states = len(A_matrix[0])
    padded_N = 2 * N - 1
    zero_pads = np.zeros(N - 1)
    kdelta = np.asarray([0] * N)
    kdelta[0] = 1

    C_matrix = np.empty((padded_N, num_states, num_states), dtype=complex)  # FFT matrix
    d_vector = np.empty((padded_N, num_states, 1), dtype=complex)  # FFT vector

    for i in range(num_states):

        d_vector[:, i, 0] = pfftw(np.append(s1b_vector[i] * s_vector[i](xs), zero_pads))

        for j in range(num_states):
            if scaled_weights is not None:
                if shifted:
                    scaled_weights[0:1] = 0
                C_matrix[:, i, j] = pfftw(np.append(S_matrix[i, j](xs) * scaled_weights, zero_pads))
            else:
                C_matrix[:, i, j] = pfftw(np.append(S_matrix[i, j](xs), zero_pads))

    AoC = A_matrix * C_matrix

    return AoC, d_vector


def continuous_prepare_AoG(A_matrix, b_vector, S_matrix, s_vector, N, xs):
    num_states = len(A_matrix[0])

    G_matrix = np.empty((N, num_states, num_states), )  # pmf matrix
    h_vecs = np.empty((N, num_states, 1), )  # pmf vector

    for i in range(num_states):
        h_vecs[:, i, 0] = b_vector[i] * s_vector[i].pdf(xs)
        for j in range(num_states):
            G_matrix[:, i, j] = S_matrix[i, j].pdf(xs)
    AoG = A_matrix * G_matrix
    return AoG, h_vecs


def continuous_prepare_gauss_seidel(A_matrix, b_vector, S_matrix, s_vector, N, xs, scaled_weights=None, left=True):
    num_states = len(A_matrix[0])
    padded_N = 2 * N - 1
    zero_pads = np.zeros(N - 1)

    delta = np.asarray([1] + ([0] * (N - 1)))
    L = np.empty((padded_N, num_states, 1), dtype=complex)  # FFT of the reciprocal of z

    dx = scaled_weights[1]

    if scaled_weights is None:
        scaled_weights = 1

    # step: create reciprocals
    for i in range(num_states):
        Q = mydeconv(delta, delta - (dx * A_matrix[i, i] * S_matrix[i, i].pdf(xs)), N)
        if left:
            L[:, i, 0] = pfftw(np.append(Q, zero_pads))  # FFT of reciprocals
        else:
            L[:, i, 0] = pfftw(np.append(Q * scaled_weights, zero_pads))  # FFT of reciprocals

    # step: deconv entire problem matrix and vector
    C_matrix = np.empty((padded_N, num_states, num_states), dtype=complex)  # FFT matrix
    d_vector = np.empty((padded_N, num_states, 1), dtype=complex)  # FFT vector

    for i in range(num_states):
        if left:
            d_vector[:, i, 0] = pfftw(np.append(b_vector[i] * s_vector[i].pdf(xs) * scaled_weights, zero_pads))
        else:
            d_vector[:, i, 0] = pfftw(np.append(b_vector[i] * s_vector[i].pdf(xs), zero_pads))

        for j in range(num_states):
            if i == j:
                if left:
                    C_matrix[:, i, j] = pfftw(np.zeros(padded_N))
                else:
                    C_matrix[:, i, j] = pfftw(np.zeros(padded_N))

                continue
            if left:
                C_matrix[:, i, j] = pfftw(np.append(S_matrix[i, j].pdf(xs) * scaled_weights, zero_pads))
            else:
                C_matrix[:, i, j] = pfftw(np.append(S_matrix[i, j].pdf(xs), zero_pads))

    # trapz convolutions
    K_matrix = A_matrix * (C_matrix * L)
    kappa_vector = (d_vector * L)

    # step: zeropad H matrix
    for i in range(num_states):
        temp_v = ipfftw(kappa_vector[:, i, 0])
        temp_v[N:] = 0
        kappa_vector[:, i, 0] = pfftw(temp_v)

        for j in range(num_states):
            temp_v = ipfftw(K_matrix[:, i, j])
            temp_v[N:] = 0
            if left:
                temp_v[:N] *= scaled_weights
                K_matrix[:, i, j] = pfftw(temp_v)
            else:
                K_matrix[:, i, j] = pfftw(temp_v)

    return K_matrix, kappa_vector


########################################################################################################################
# Solvers
def continuous_power_romberg_no_pdf_ws1(a, b, A_matrix, s1_b_vector, S_matrix, s_vector, N,
                                        chosen_state=0, level=1,
                                        use_trapz=True):
    # remember: AoG can be done more efficiently by computing the highest level version only
    old_N = N
    sols = np.empty(level, dtype=np.ndarray)

    # remember: solve each, then collapse
    for i in range(level):
        if i > 0:
            N = 2 * N - 1
        xs = np.linspace(a, b, N)
        dx = (b - a) / (N - 1)
        scaled_w = dx * np.asarray([1] * N)
        scaled_w[0] = 0

        AoC_w, d_v = continuous_prepare_AoC_no_pdf_w_s1(A_matrix, s1_b_vector, S_matrix, s_vector, N, xs,
                                                        scaled_weights=scaled_w.copy())
        if use_trapz:
            AoC, _ = continuous_prepare_AoC_no_pdf_w_s1(A_matrix, s1_b_vector, S_matrix, s_vector, N, xs,
                                                        scaled_weights=None)
        else:
            AoC = None

        pdfx_trapz = continuous_power_method_trapz_exact(AoC, AoC_w, d_v, N, scaled_w.copy(), max_iter=567,
                                                         trapz=use_trapz)

        sols[i] = pdfx_trapz[chosen_state, :N]

    return romberg_helper(level, old_N, sols)


def continuous_power_romberg_no_pdf(a, b, A_matrix, b_vector, S_matrix, s_vector, N,
                                    chosen_state=0, level=1,
                                    use_trapz=True):
    # remember: AoG can be done more efficiently by computing the highest level version only
    old_N = N
    sols = np.empty(level, dtype=np.ndarray)

    # remember: solve each, then collapse
    for i in range(level):
        if i > 0:
            N = 2 * N - 1
        xs = np.linspace(a, b, N)
        dx = (b - a) / (N - 1)
        scaled_w = dx * np.asarray([1] * N)
        scaled_w[0] = 0

        AoC_w, d_v = continuous_prepare_AoC_no_pdfs(A_matrix, b_vector, S_matrix, s_vector, N, xs,
                                                    scaled_weights=scaled_w.copy())

        if use_trapz:
            AoC, _ = continuous_prepare_AoC_no_pdfs(A_matrix, b_vector, S_matrix, s_vector, N, xs, scaled_weights=None)
        else:
            AoC = None

        pdfx_trapz = continuous_power_method_trapz_exact(AoC, AoC_w, d_v, N, scaled_w.copy(), max_iter=567,
                                                         trapz=use_trapz)
        sols[i] = pdfx_trapz[chosen_state, :N]

    return romberg_helper(level, old_N, sols)


def romberg_helper(level, old_N, sols):
    for i in range(level):
        M = old_N
        for j in range(level - (i + 1)):
            for k in range(M):
                denom_term = 4 ** (i + 1) - 1
                sols[j][k] = ((1 + denom_term) / (denom_term) * sols[j + 1][2 * k]) - (1 / denom_term * sols[j][k])
            M = 2 * M - 1
    return sols[0]


def continuous_power_method_trapz_exact(AoC, AoC_w, d_vector, N, scaled_weights, max_iter=10000, x=None, trapz=True,
                                        atol=1e-16):
    padded_N = 2 * N - 1
    num_states = len(d_vector[0, :, 0])

    scaled_weights = np.asarray(np.append(scaled_weights, np.zeros(N - 1)))

    pmfx = np.zeros((padded_N, num_states), dtype=np.complex128).T

    if x is None:
        x = np.zeros((padded_N, num_states, 1),
                     dtype=np.complex128)  # remember: should be zeroes, delta leads to ones, zeroes lead to zeroes.
    x2 = x.copy()

    count = 1
    for i in range(1, max_iter):

        if trapz:
            new_x = 0.5 * (np.matmul(AoC_w, x) + np.matmul(AoC, x2)) + d_vector
        else:
            new_x = (np.matmul(AoC_w, x)) + d_vector

        new_pmfx = np.real(np.asarray([ipfftw(breadth) for breadth in new_x[..., 0].T]))
        new_pmfx[:, N:] = 0

        if alternative_allclose(new_pmfx.T, pmfx.T, N, atol):
            print("Total Iterations:", count)
            return new_pmfx
        else:
            # remember:use new_pmfx to define x2 also
            x = np.asarray([pfftw(breadth) for breadth in new_pmfx]).T[..., np.newaxis]
            if trapz:
                x2 = np.asarray([pfftw(scaled_weights * breadth) for breadth in new_pmfx]).T[..., np.newaxis]
            pmfx = new_pmfx

        count += 1
    print("Total Iterations:", count)
    return np.real(pmfx)


def continuous_power_method_trapz_approx(AoC, AoC_w, d_vector, N, scaled_weights, max_iter=10000, x=None, trapz=True,
                                         atol=1e-16):
    padded_N = 2 * N - 1
    num_states = len(d_vector[0, :, 0])

    scaled_weights = np.asarray(np.append(scaled_weights, np.zeros(N - 1)))

    if x is None:
        x = np.zeros((padded_N, num_states, 1),
                     dtype=np.complex128)
    x2 = x.copy()
    new_x = x.copy()

    count = 1
    for i in range(1, max_iter):
        if trapz:
            new_x = 0.5 * (np.matmul(AoC_w, x) + np.matmul(AoC, x2)) + d_vector
        else:
            new_x = (np.matmul(AoC_w, x)) + d_vector

        if alternative_allclose(new_x, x, N, atol=atol):
            print("Total Iterations:", count)
            return np.real(np.asarray([ipfftw(breadth) for breadth in new_x[..., 0].T]))
        else:

            x = new_x
            if trapz:
                new_pmfx = np.real(np.asarray([ipfftw(breadth) for breadth in new_x[..., 0].T]))
                x2 = np.asarray([pfftw(scaled_weights * breadth) for breadth in new_pmfx]).T[..., np.newaxis]
        count += 1
    print("Total Iterations:", count)
    return np.real(np.asarray([ipfftw(breadth) for breadth in new_x[..., 0].T]))


def count_zeroes(l):
    count = 0
    for i in range(len(l)):
        if l[i] <= 1e-16 or l[i] < 0:  # second check is redundant
            count += 1
        else:
            return count
    assert False, "zero vector found: " + str(l)


def mydeconv(f, g, N):
    zs = count_zeroes(g)
    extra_zeroes = np.zeros(2 * N)
    return np.polydiv(np.append(f, extra_zeroes), g[zs:])[0][:N]


def alternative_allclose(A, B, n_points,
                         atol=1e-16):  # warn: debugging at atol = 1e-10, IMPORTANT FOR CONTINUOUS CASE!
    v = np.abs(A[:n_points, :] - B[:n_points, :]).max()
    assert v <= 1e2, "Probably diverging, stopping algorithm"
    return v <= atol
