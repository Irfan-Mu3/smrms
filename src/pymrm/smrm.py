import random as rd
import time
import numpy as np
import pyfftw as fftw
from numba import jit
from scipy.signal import deconvolve

from pymrm.util import alternative_allclose

fftw.interfaces.cache.enable()
pfftw = fftw.interfaces.numpy_fft.fft
ipfftw = fftw.interfaces.numpy_fft.ifft

"""
Solves smrm reachability problems: The probability of reaching a set of states B, with reward cumulated r= 0,1,...,N-1,
is derived. Available solvers are: GE, power method, Jacobi, and Gauss-Siedel. We recommend using the power method.
Two approximate algorithms are available: approximate power, and approximate GE (which uses LU decomp. via np.solve).

To see how an sMRM is created, please see smrm_usecase_examples.py. The simplest case is to use the power method
 with the following data: A ,b , S_matrix and s_vector, N, xs. 
 
The solvers are two parts: preparation then solving. The file smrm_usecase_examples.py details which
preparation method is related to which solver.

Theory underpinning these techniques can be found in the thesis:
Algorithms for reachability problems on stochastic Markov reward models, Irfan Muhammad, University of Birmingham.

Notation:
A(s,t) - Pr(s,t). Defined for s,t in (S_?)**2.
b(s) - Pr(s,B). Defined for s in S_?.
S_matrix - This stores the reward random variables matrix, and can be used to derive G.
s_vector - This stores the reward r.v. vector, and can be used to derive h.
G_matrix (or just G) - The hypermatrix of reward random variable values.
h_vector (or h_vecs) - The hypervector of reward random variable values.
C_matrix - This is the fftw of G.
d_vector - This is the fftw of h.
AoG - the hadamard product between A and G.
AoC - the hadamard product between A and the fftw transform of G, which is C.
I_AoG denotes I - AoG, where I is the hypermatrix identity, i.e. matrix of kronecker deltas.

K_matrix - The hypermatrix used for Gauss-Siedel iterations. This is derived via AoG and h_vec.
Kappa_vector - The hypervector used for Gauss-Siedel iterations.
"""


########################################################################################################################
# Preparations

def prepare_power(A_matrix, b_vector, S_matrix, s_vector, N, xs, padding=None):
    """
    :param A_matrix: Square probability transition matrix. Pr(s,t) - The probability of transitioning between s and t.
        This has size (S?,S?)
    :param b_vector: Probability vector, Pr(s,B) - This is the probability of entering B directly from each s. The dim
        of this vector has to be equal to that of A_matrix. This has size (S?,1)
    :param S_matrix: Matrix of random variables, of which has a .pmf() call to obtain values. Size is equal to
        A_matrix.
    :param s_vector:  Vector of random variables, of which of which has a .pmf() call to obtain values.
        Size is equal to b_vector.
    :param N: The number of points needed from the cumulated reward distribution.
    :param xs: This is typically np.arange(0,N). Other equidistant lattices are possible. Must be of length N.
    :param padding: This is only necessary if using the approximate power method.
    :return: Returns the hypermatrix AoC and the hypervector d_vector.
    """
    num_states = len(A_matrix[0])
    if padding is None:
        padded_N = 2 * N - 1
        zero_pads = np.zeros(N - 1)
    else:
        padded_N = N + padding
        zero_pads = np.zeros(padding)

    C_matrix = np.empty((padded_N, num_states, num_states), dtype=complex)  # FFT matrix
    d_vector = np.empty((padded_N, num_states, 1), dtype=complex)  # FFT vector

    for i in range(num_states):
        d_vector[:, i, 0] = pfftw(np.append(b_vector[i] * s_vector[i].pmf(xs), zero_pads))
        for j in range(num_states):
            C_matrix[:, i, j] = pfftw(np.append(S_matrix[i, j].pmf(xs), zero_pads))

    AoC = A_matrix * C_matrix
    return AoC, d_vector


def prepare_power_pmfs(A_matrix, b_vector, pmf_matrix, pmf_vector, N, padding=None):
    """
    :param A_matrix: Square probability transition matrix. Pr(s,t) - The probability of transitioning between s and t.
        This has size (S?,S?)
    :param b_vector: Probability vector, Pr(s,B) - This is the probability of entering B directly from each s. The dim
        of this vector has to be equal to that of A_matrix. This has size (S?,1)
    :param S_matrix: Hypermatrix of random variable values. Has size (N, S?, S?).
    :param s_vector:  Hypervector of random variable values. Has size (N, S?, 1).
    :param N: The number of points needed from the cumulated reward distribution.
    :param xs: This is typically np.arange(0,N). Other equidistant lattices are possible. Must be of length N.
    :param padding: This is only necessary if using the approximate power method.
    :return: Returns the hypermatrix AoC and the hypervector d_vector.
    """
    num_states = len(A_matrix[0])
    if padding is None:
        padded_N = 2 * N - 1
        zero_pads = np.zeros(N - 1)
    else:
        padded_N = N + padding
        zero_pads = np.zeros(padding)

    C_matrix = np.empty((padded_N, num_states, num_states), dtype=complex)  # FFT matrix
    d_vector = np.empty((padded_N, num_states, 1), dtype=complex)  # FFT vector

    for i in range(num_states):
        d_vector[:, i, 0] = pfftw(np.append(b_vector[i] * pmf_vector[i], zero_pads))
        for j in range(num_states):
            C_matrix[:, i, j] = pfftw(np.append(pmf_matrix[i, j], zero_pads))

    AoC = A_matrix * C_matrix

    return AoC, d_vector


def prepare_ge(A_matrix, b_vector, S_matrix, s_vector, N, xs):
    num_states = len(A_matrix[0])

    G_matrix = np.empty((N, num_states, num_states), dtype=np.float32)  # pmf matrix
    h_vecs = np.empty((N, num_states, 1), dtype=np.float32)  # pmf vector

    for i in range(num_states):
        h_vecs[:, i, 0] = b_vector[i] * s_vector[i].pmf(xs)
        for j in range(num_states):
            G_matrix[:, i, j] = S_matrix[i, j].pmf(xs)

    AoG = A_matrix * G_matrix

    I_matrix = np.eye(num_states, num_states)[np.newaxis, ...]
    arrays = (I_matrix, np.zeros((N - 1, num_states, num_states)))
    I_AoG = np.vstack(arrays) - AoG

    return I_AoG, h_vecs


def prepare_ge_pmfs(A_matrix, b_vector, pmf_matrix, pmf_vector, N):
    num_states = len(A_matrix[0])

    G_matrix = np.empty((N, num_states, num_states), dtype=np.float32)  # pmf matrix
    h_vecs = np.empty((N, num_states, 1), dtype=np.float32)  # pmf vector

    for i in range(num_states):
        h_vecs[:, i, 0] = b_vector[i] * pmf_vector[i]
        for j in range(num_states):
            G_matrix[:, i, j] = pmf_matrix[i, j]

    AoG = A_matrix * G_matrix

    I_matrix = np.eye(num_states, num_states)[np.newaxis, ...]
    arrays = (I_matrix, np.zeros((N - 1, num_states, num_states)))
    I_AoG = np.vstack(arrays) - AoG

    return I_AoG, h_vecs


def prepare_gs(A_matrix, b_vector, S_matrix, s_vector, N, xs):
    num_states = len(A_matrix[0])
    padded_N = 2 * N - 1
    zero_pads = np.zeros(N - 1)

    kronecker_delt = np.asarray([1] + ([0] * (N - 1)))
    L = np.empty((padded_N, num_states, 1), dtype=complex)  # FFT of the reciprocal of z

    # step: create reciprocals
    for i in range(num_states):
        temp_v = kronecker_delt - (A_matrix[i, i] * S_matrix[i, i].pmf(xs))
        reciprocal, _ = deconvolve(np.append(kronecker_delt, zero_pads), temp_v)
        L[:, i, 0] = pfftw(np.append(reciprocal, zero_pads))  # FFT of reciprocals

    # step: deconv entire problem matrix and vector
    C_matrix = np.empty((padded_N, num_states, num_states), dtype=complex)  # FFT matrix
    d_vector = np.empty((padded_N, num_states, 1), dtype=complex)  # FFT vector

    for i in range(num_states):
        d_vector[:, i, 0] = pfftw(np.append(b_vector[i] * s_vector[i].pmf(xs), zero_pads))
        for j in range(num_states):
            if i == j:
                C_matrix[:, i, j] = pfftw(np.zeros(padded_N))
                continue
            C_matrix[:, i, j] = pfftw(np.append(S_matrix[i, j].pmf(xs), zero_pads))

    K_matrix = A_matrix * C_matrix * L
    kappa_vector = d_vector * L

    # step: zeropad H matrix
    for i in range(num_states):
        temp_v = ipfftw(kappa_vector[:, i, 0])
        temp_v[N:] = 0
        kappa_vector[:, i, 0] = pfftw(temp_v)

        for j in range(num_states):
            temp_v = ipfftw(K_matrix[:, i, j])
            temp_v[N:] = 0
            K_matrix[:, i, j] = pfftw(temp_v)

    return K_matrix, kappa_vector


def prepare_gs_pmfs(A_matrix, b_vector, pmf_matrix, pmf_vector, N):
    num_states = len(A_matrix[0])
    padded_N = 2 * N - 1
    zero_pads = np.zeros(N - 1)

    kronecker_delt = np.asarray([1] + ([0] * (N - 1)))
    L = np.empty((padded_N, num_states, 1), dtype=complex)  # FFT of the reciprocal of z

    # step: create reciprocals
    for i in range(num_states):
        temp_v = kronecker_delt - (A_matrix[i, i] * pmf_matrix[i, i])
        reciprocal, _ = deconvolve(np.append(kronecker_delt, zero_pads), temp_v)
        L[:, i, 0] = pfftw(np.append(reciprocal, zero_pads))  # FFT of reciprocals

    # step: deconv entire problem matrix and vector
    C_matrix = np.empty((padded_N, num_states, num_states), dtype=complex)  # FFT matrix
    d_vector = np.empty((padded_N, num_states, 1), dtype=complex)  # FFT vector

    for i in range(num_states):
        d_vector[:, i, 0] = pfftw(np.append(b_vector[i] * pmf_vector[i], zero_pads))
        for j in range(num_states):
            if i == j:
                C_matrix[:, i, j] = pfftw(np.zeros(padded_N))
                continue
            C_matrix[:, i, j] = pfftw(np.append(pmf_matrix[i, j], zero_pads))

    K_matrix = A_matrix * C_matrix * L
    kappa_vector = d_vector * L

    # step: zeropad H matrix
    for i in range(num_states):
        temp_v = ipfftw(kappa_vector[:, i, 0])
        temp_v[N:] = 0
        kappa_vector[:, i, 0] = pfftw(temp_v)

        for j in range(num_states):
            temp_v = ipfftw(K_matrix[:, i, j])
            temp_v[N:] = 0
            K_matrix[:, i, j] = pfftw(temp_v)

    return K_matrix, kappa_vector


def create_AoG_h(A_matrix, b_vector, S_matrix, s_vector, N, xs):
    num_states = len(A_matrix[0])

    G_matrix = np.empty((N, num_states, num_states), dtype=np.float32)  # pmf matrix
    h_vecs = np.empty((N, num_states, 1), dtype=np.float32)  # pmf vector

    for i in range(num_states):
        h_vecs[:, i, 0] = b_vector[i] * s_vector[i].pmf(xs)
        for j in range(num_states):
            G_matrix[:, i, j] = S_matrix[i, j].pmf(xs)

    AoG = A_matrix * G_matrix
    return AoG, h_vecs


def create_I_AoC_matrix(AoC: np.ndarray, num_states: int):
    diagonal_deltas = np.eye(num_states, num_states)[np.newaxis, ...]
    diag_deltas = np.repeat(diagonal_deltas, AoC.shape[0], axis=0)  # padded
    return diag_deltas - AoC


def create_combined_mat_for_gs(I_AoG, h_v):
    return np.dstack((I_AoG, h_v))


########################################################################################################################
# Solvers

def mydeconv(f, g, N):
    extra_zeros = np.zeros(N - 1)
    return np.polydiv(np.append(f, extra_zeros), g)[0][:N]


def myconv(f, g, N):
    extra_zeros = np.zeros(N - 1)
    F = pfftw(np.append(f, extra_zeros))
    G = pfftw(np.append(g, extra_zeros))
    return np.real(ipfftw(F * G))[:N]


def solve_ge(I_AoG, h_vecs, N):
    num_states = len(I_AoG[0])
    for k in range(num_states - 1):
        for i in range(k + 1, num_states):

            I_AoG[:, i, k] = mydeconv(I_AoG[:, i, k], I_AoG[:, k, k], N)

            for j in range(k + 1, num_states):
                I_AoG[:, i, j] -= myconv(I_AoG[:, k, j], I_AoG[:, i, k], N)

    for k in range(num_states - 1):
        for i in range(k + 1, num_states):
            h_vecs[:, i, 0] = h_vecs[:, i, 0] - myconv(h_vecs[:, k, 0], I_AoG[:, i, k], N)

    pmfx = h_vecs[..., -1].T

    for i in range(num_states - 1, -1, -1):
        for j in range(i + 1, num_states):
            pmfx[i] -= myconv(I_AoG[:, i, j], pmfx[j], N)
        pmfx[i] = mydeconv(pmfx[i], I_AoG[:, i, i], N)
    return np.real(pmfx)


def solve_ge_opt(I_AoG, h_vecs, N):
    num_states = len(I_AoG[0])
    delta = np.zeros(N)
    delta[0] = 1

    for k in range(num_states - 1):
        Q = mydeconv(delta, I_AoG[:, k, k], N)

        for i in range(k + 1, num_states):
            I_AoG[:, i, k] = myconv(I_AoG[:, i, k], Q, N)

            for j in range(k + 1, num_states):
                I_AoG[:, i, j] -= myconv(I_AoG[:, k, j], I_AoG[:, i, k], N)

            h_vecs[:, i, 0] = h_vecs[:, i, 0] - myconv(h_vecs[:, k, 0], I_AoG[:, i, k], N)

    pmfx = h_vecs[..., -1].T
    for i in range(num_states - 1, -1, -1):
        for j in range(i + 1, num_states):
            pmfx[i] -= myconv(I_AoG[:, i, j], pmfx[j], N)
        pmfx[i] = mydeconv(pmfx[i], I_AoG[:, i, i], N)

    return np.real(pmfx)


def solve_ge2(AA, N):
    num_states = len(AA[0, :, 0])
    for k in range(num_states - 1):
        for i in range(k + 1, num_states):
            AA[:, i, k] = mydeconv(AA[:, i, k], AA[:, k, k], N)
            for j in range(k + 1, num_states + 1):
                AA[:, i, j] -= myconv(AA[:, k, j], AA[:, i, k], N)

    pmfx = AA[:, ..., -1].T

    for i in range(num_states - 1, -1, -1):
        for j in range(i + 1, num_states):
            pmfx[i] -= myconv(AA[:, i, j], pmfx[j], N)
        pmfx[i] = mydeconv(pmfx[i], AA[:, i, i], N)

    return np.real(pmfx)


def solve_ge_AoG(AoG, h_vecs, N):
    num_states = len(AoG[0])
    delta = np.zeros(N)
    delta[0] = 1
    AoG *= -1

    for k in range(num_states - 1):
        for i in range(k + 1, num_states):
            AoG[:, i, k] = mydeconv(AoG[:, i, k], delta + AoG[:, k, k], N)

            for j in range(k + 1, num_states):
                AoG[:, i, j] -= myconv(AoG[:, k, j], AoG[:, i, k], N)

            h_vecs[:, i, 0] = h_vecs[:, i, 0] - myconv(h_vecs[:, k, 0], AoG[:, i, k], N)
    pmfx = h_vecs[..., -1].T

    for i in range(num_states - 1, -1, -1):
        for j in range(i + 1, num_states):
            pmfx[i] -= myconv(AoG[:, i, j], pmfx[j], N)
        pmfx[i] = mydeconv(pmfx[i], delta + AoG[:, i, i], N)

    return np.real(pmfx)


def solve_power(AoC, d_vector, N, max_iter=10000, x=None, force_full_iter=False, eps=1e-16):
    padded_N = 2 * N - 1
    num_states = len(d_vector[0, :, 0])
    pmfx = np.zeros((padded_N, num_states), dtype=np.complex128).T
    if x is None:
        x = np.zeros((padded_N, num_states, 1),
                     dtype=np.complex128)  # remember: should be zeroes, delta leads to ones, zeroes lead to zeroes.
    count = 1
    for i in range(1, max_iter):

        new_x = np.matmul(AoC, x) + d_vector
        new_pmfx = np.real(np.asarray([ipfftw(breadth) for breadth in new_x[..., 0].T]))
        new_pmfx[:, N:] = 0

        if alternative_allclose(new_pmfx.T, pmfx.T, N, atol=eps) and not force_full_iter:
            print("Total Iterations:", count)
            return new_pmfx, count
        else:
            x = np.asarray([pfftw(breadth) for breadth in new_pmfx]).T[..., np.newaxis]
            pmfx = new_pmfx

        count += 1
    print("Total Iterations:", count)
    return np.real(pmfx), count


@jit
def myconvolve_direct(f, g, N):
    sol = np.zeros(N)
    for i in range(N):
        temp = 0
        for j in range(0, i):
            temp += f[j] * g[i - j]
        sol[i] = temp
    return sol


def convolve_matrix_op(AoG, pmf_vecs, h_vecs):
    # warn: need to experiment with AoG as sparse
    N = len(h_vecs[:, 0, 0])
    num_states = len(h_vecs[0, :, 0])

    sol_hypvec = np.zeros((N, num_states)).T
    for i in range(num_states):
        temp_sol = np.zeros(N)
        for j in range(num_states):
            # remember: should return length N
            temp_sol += myconvolve_direct(AoG[:, i, j], pmf_vecs[j], N)
        sol_hypvec[i] = temp_sol + h_vecs[:, i, 0]
    return sol_hypvec


def solve_power_slow(AoG, h_vecs, N, max_iter=1000):
    num_states = len(h_vecs[0, :, 0])
    pmfx = np.zeros((N, num_states)).T
    count = 1
    for i in range(1, max_iter):
        new_pmfx = convolve_matrix_op(AoG, pmfx, h_vecs)
        if alternative_allclose(new_pmfx, pmfx, N):
            print("Total Iterations:", count)
            return new_pmfx, count
        else:
            pmfx = new_pmfx

        count += 1
    print("Total Iterations:", count)
    return pmfx, count


def solve_power_approx(AoC, d_vector, T, max_iter=1000, x=None,
                       eps=1e-7):
    num_states = len(d_vector[0, :, 0])
    if x is None:
        x = np.zeros((T, num_states, 1),
                     dtype=np.complex128)

    new_x = np.zeros_like(x)
    count = 1
    for _ in range(1, max_iter):
        new_x = np.matmul(AoC, x) + d_vector

        if alternative_allclose(np.real(new_x), np.real(x), T, atol=eps) \
                and alternative_allclose(np.imag(new_x), np.imag(x), T, atol=eps):
            print("Total Iterations:", count)
            new_pmfx = np.real(np.asarray([ipfftw(breadth) for breadth in new_x[..., 0].T]))
            return new_pmfx, count
        else:
            x = new_x
        count += 1

    new_pmfx = np.real(np.asarray([ipfftw(breadth) for breadth in new_x[..., 0].T]))
    return new_pmfx, count


def solve_gs(AoC, d_vector, N, max_iter=10000, x=None, eps=1e-16):
    padded_N = 2 * N - 1
    num_states = len(d_vector[0, :, 0])

    triu_nk = np.triu(AoC, 1)  # strictly above the diagonal
    tril_nk = np.tril(AoC, -1)  # strictly below the diagonal

    x = np.zeros((padded_N, num_states, 1), dtype=np.complex128)
    pmfx = np.zeros((num_states, padded_N), dtype=np.complex128)

    count = 1
    for i in range(1, max_iter):

        temp_x = np.matmul(triu_nk, x) + d_vector
        new_x = np.zeros((padded_N, num_states, 1), dtype=np.complex128)
        new_pmfx = np.zeros((num_states, padded_N), dtype=np.complex128)

        # remember: solve for i== 0
        new_x[:, 0, 0] += temp_x[:, 0, 0]
        new_pmfx[0, :] = np.real(ipfftw(new_x[:, 0, 0].T))
        new_pmfx[0, N:] = 0
        new_x[:, 0, 0] = pfftw(new_pmfx[0])

        for j in range(1, num_states):  # length of A (a square matrix)
            new_x[:, j, 0] += temp_x[:, j, 0]
            new_x[:, j] += (tril_nk[:, j, :j][:, np.newaxis, :] @ new_x[:, :j])[..., 0]

            new_pmfx[j, :] = np.real(ipfftw(new_x[:, j, 0].T))
            new_pmfx[j, N:] = 0
            new_x[:, j, 0] = pfftw(new_pmfx[j])

        if alternative_allclose(new_pmfx.T, pmfx.T, N, atol=eps):
            print("Total Iterations:", count)
            return np.real(new_pmfx), count
        else:
            x = new_x
            pmfx = new_pmfx

        count += 1

    print("Total Iterations:", count)
    return np.real(pmfx), count


def solve_lu_decomp_approx(I_AoC, d_vector):
    x = np.linalg.solve(I_AoC, d_vector)
    pmfx = np.real(np.asarray([ipfftw(breadth) for breadth in x[..., 0].T]))
    return pmfx


def sample_smrm(num_states, xs, sampling_P_matrix, S_matrix, s_vector, N, max_iter=1000, init_state=0):
    states = np.arange(num_states + 1)
    histograms = np.zeros(N + 1)

    start = time.time()
    for _ in range(max_iter):
        accum_reward = 0
        curr_state = init_state
        while True:
            new_curr_state = rd.choices(states, sampling_P_matrix[curr_state, :])[0]

            if new_curr_state != num_states:  # if the curr state is not the absorbing state
                accum_reward += S_matrix[curr_state, new_curr_state].rvs(1)
            else:
                accum_reward += s_vector[curr_state].rvs(1)
            if accum_reward >= xs[-1]:
                histograms[N - 1] += 1
                break
            if new_curr_state == num_states:
                histograms[accum_reward] += 1
                break

            curr_state = new_curr_state

    print("Time for sampling:", time.time() - start)
    return histograms
