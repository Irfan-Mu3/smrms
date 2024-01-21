import matplotlib.pyplot as plt
import numpy as np
import sparse
import seaborn as sns


def generate_random_mc(num_states, uniform=False, max_prob=None):
    if not uniform:
        P_matrix = np.random.random((num_states + 1, num_states))  # state transition probability matrix
    else:
        P_matrix = np.ones((num_states + 1, num_states))

    P_matrix += 0.01  # to prevent columns summing to zero, which introduces nan in the next step (not neccessary)
    P_matrix = (P_matrix / P_matrix.sum(0)).T

    if max_prob is not None:  # used to create small probabilities, or rare-event MCs.
        P_matrix *= max_prob

    b_vector = P_matrix[:, -1]  # use last column as probability of entering B
    A_matrix = P_matrix[:, :-1]  # with cardinality (S_? x S_?)

    # P_matrix is used for sampling solution
    return A_matrix, b_vector, P_matrix


def generate_random_mc_block_partitions(num_states, num_pass=200, block_scale=5):
    P_matrix = np.zeros((num_states, num_states + 1))
    full_state_idxs = np.arange(num_states)

    # for each state, create probability that it can reach B
    reach_idxs = np.random.choice(full_state_idxs, int(num_states * np.random.rand()), replace=False)
    reach_probs = np.random.rand(len(reach_idxs))
    P_matrix[reach_idxs, -1] = reach_probs

    # perform n pass
    for _ in range(num_pass):
        block_size = int((np.random.rand() * num_states) / (2 * block_scale))
        initial_idx = np.maximum((np.random.rand(2) * num_states).astype(int) - block_size, [0, 0])
        P_matrix[initial_idx[0]:initial_idx[0] + 2 * block_size,
        initial_idx[1]:initial_idx[1] + 2 * block_size] += 1

    row_sums = P_matrix.sum(0)  # for each col, sum up all the rows
    non_zero_idxs = row_sums.nonzero()
    P_matrix[:, non_zero_idxs] /= row_sums[non_zero_idxs]

    b_vector = P_matrix[:, -1]  # use last column as probability of entering B
    A_matrix = P_matrix[:, :-1]  # with cardinality (S_? x S_?)

    return A_matrix, b_vector, P_matrix


def generate_random_mc_block(num_states, num_pass=200, block_scale=5):
    P_matrix = np.zeros((num_states, num_states + 1))
    full_state_idxs = np.arange(num_states)

    # for each state, create probability that it can reach B
    reach_idxs = np.random.choice(full_state_idxs, int(num_states * np.random.rand()), replace=False)
    reach_probs = np.random.rand(len(reach_idxs))
    P_matrix[reach_idxs, -1] = reach_probs

    # perform n pass
    for _ in range(num_pass):
        block_size = int((np.random.rand() * num_states) / (2 * block_scale))
        initial_idx = np.maximum((np.random.rand(2) * num_states).astype(int) - block_size, [0, 0])
        P_matrix[initial_idx[0]:initial_idx[0] + 2 * block_size,
        initial_idx[1]:initial_idx[1] + 2 * block_size] += 1

    row_sums = P_matrix.sum(0)  # for each col, sum up all the rows
    non_zero_idxs = row_sums.nonzero()
    P_matrix[:, non_zero_idxs] /= row_sums[non_zero_idxs]

    b_vector = P_matrix[:, -1]  # use last column as probability of entering B
    A_matrix = P_matrix[:, :-1]  # with cardinality (S_? x S_?)

    return A_matrix, b_vector, P_matrix


def generate_random_mc_npass(num_states, num_pass=1000):
    P_matrix = np.zeros((num_states, num_states + 1))
    full_state_idxs = np.arange(num_states)

    # for each state, create probabilty that it can reach B
    reach_idxs = np.random.choice(full_state_idxs, int(num_states * np.random.rand()), replace=False)
    reach_probs = np.random.rand(len(reach_idxs))
    P_matrix[reach_idxs, -1] = reach_probs

    # perform n pass
    for _ in range(num_pass):
        temp = np.random.uniform(0, 1 - P_matrix.sum(1), num_states)  # a column of probabilities
        choices = np.random.choice(reach_idxs, num_states, replace=True)
        P_matrix[full_state_idxs, choices] += temp
        reach_idxs = full_state_idxs

    b_vector = P_matrix[:, -1]  # get last column
    A_matrix = P_matrix[:, :-1]  # cluster matrix A

    return A_matrix, b_vector, P_matrix


def generate_random_mc_npassrand(num_states, num_pass=1000):
    P_matrix = np.zeros((num_states, num_states + 1))
    full_state_idxs = np.arange(num_states)

    # for each state, create probabilty that it can reach B
    reach_idxs = np.random.choice(full_state_idxs, np.maximum(1, int(num_states * np.random.rand())), replace=False)
    reach_probs = np.random.rand(len(reach_idxs))
    P_matrix[reach_idxs, -1] = reach_probs

    # perform n pass
    for _ in range(num_pass):
        sel_states = np.random.permutation(full_state_idxs)
        choices = np.random.choice(reach_idxs, num_states, replace=True)
        temp = np.random.uniform(0, 1 - P_matrix[sel_states, :].sum(1), len(sel_states))  # a column of probabilities
        P_matrix[sel_states, choices] += temp
        reach_idxs = sel_states

    b_vector = P_matrix[:, -1]  # get last column
    A_matrix = P_matrix[:, :-1]  # cluster matrix A

    return A_matrix, b_vector, P_matrix


def generate_random_mc_sparse(num_states, density=0.1):
    P_matrix = sparse.random((num_states + 1, num_states), density)  # state transition probability matrix
    P_matrix = P_matrix.todense()
    full_state_idxs = np.arange(num_states)

    # for each state, create probabilty that it can reach B
    reach_idxs = np.random.choice(full_state_idxs, np.maximum(1, int(num_states * np.random.rand())), replace=False)
    reach_probs = np.random.rand(len(reach_idxs))
    P_matrix[reach_idxs, -1] = reach_probs

    # P_matrix[-1, :] = np.random.rand(1, num_states) + 0.0001  # to ensure reachability

    row_sums = P_matrix.sum(0)  # for each col, sum up all the rows
    non_zero_idxs = row_sums.nonzero()
    P_matrix[:, non_zero_idxs] /= row_sums[non_zero_idxs]

    P_matrix = P_matrix.T

    b_vector = P_matrix[:, -1]  # get last column
    A_matrix = P_matrix[:, :-1]  # cluster matrix A

    return A_matrix, b_vector, P_matrix

# if __name__ == "__main__":
#     num_states = 30
#     res = generate_random_MC_npassrand(num_states)
#     res4 = generate_random_MC_block(num_states)
#     res3 = generate_random_MC(num_states)
#     res2 = generate_random_MC_sparse(num_states)
#
#     fig, axs = plt.subplots(nrows=2, ncols=2, )
#
#     # rasterization is needed to remove lines from pdfs
#     sns.heatmap(res[2], ax=axs[0, 0], rasterized=True)
#     sns.heatmap(res4[2], ax=axs[0, 1], rasterized=True)
#     sns.heatmap(res3[2], ax=axs[1, 0], rasterized=True)
#     sns.heatmap(res2[2], ax=axs[1, 1], rasterized=True)
#
#     axs[0, 0].set_title('N-pass MC')
#     axs[0, 1].set_title('Block MC')
#     axs[1, 0].set_title('Uniform MC')
#     axs[1, 1].set_title('Sparse MC')
#
#     plt.show()
