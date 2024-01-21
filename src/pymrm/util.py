import numpy as np
from numba import njit


def generate_random_MC(num_states, uniform=False, max_prob=None):
    if not uniform:
        P_matrix = np.random.random((num_states + 1, num_states))  # state transition probability matrix
    else:
        P_matrix = np.ones((num_states + 1, num_states))

    P_matrix += 0.01  # remember: to prevent columns summing to zero, which introduces nan in the next step
    P_matrix = (P_matrix / P_matrix.sum(0)).T

    if max_prob is not None:  # used to create small probabilities, or rare-event MCs.
        P_matrix *= max_prob

    b_vector = P_matrix[:, -1]  # get last column
    A_matrix = P_matrix[:, :-1]  # cluster matrix A

    # P_matrix is used for sampling solution
    return A_matrix, b_vector, P_matrix

@njit
def alternative_allclose(A, B, n_points,
                         atol=1e-16):  # warn: debugging at atol = 1e-10, IMPORTANT FOR CONTINUOUS CASE!
    v = np.abs(A[:n_points, :] - B[:n_points, :]).max()

    assert v <= 1e2, "Probably diverging, stopping algorithm"
    return v <= atol

