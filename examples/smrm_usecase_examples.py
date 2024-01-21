import scipy.stats as st
import matplotlib.pyplot as plt
from pymrm.smrm import *
from pymrm.util import generate_random_MC

"""
Use-cases for the various solvers.
"""

if __name__ == "__main__":
    np.random.seed(123456)  # seed

    ########################################################
    # create a toy smrm

    # N is the number of points we will solve for each state.
    num_states, N = 3, 1024

    # Generate random markov chain. A_matrix is size (num_states,num_states), b_vector has size (num_states,1).
    # The sampling matrix can be used for the sample_smrm method later.
    A_matrix, b_vector, sampling_P_matrix = generate_random_MC(num_states)

    # The lattice points (r = 0,1,...,N-1)
    xs = np.arange(0, N)

    # Create the matrix storing the reward random variables for each transition in A_matrix (S_matrix)
    # and b_vector (s_vector).
    S_matrix = np.empty((num_states, num_states), dtype=object)
    s_vector = np.empty(num_states, dtype=object)

    # Create random random variables. Note, this example requires these random variables having a .pmf() call.
    # Alternatively, the values xs can be sampled for each rv and used instead (see smrm.py for the alt. preparation
    # methods.
    for i in range(num_states):
        s_vector[i] = st.binom(N, np.random.rand())
        for j in range(num_states):
            S_matrix[i, j] = st.poisson(np.random.rand())
    ########################################################

    # prepare biplots. (We plot only the results from the first state (out of 3)).
    # We plot the actual pmfs, and the errors relative to the power method (absolute difference).
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # solve via the power method (first prepare, then solve).
    AoC, d_v = prepare_power(A_matrix, b_vector, S_matrix, s_vector, N, xs)
    pmfx_power, c = solve_power(AoC, d_v, N, max_iter=1000)
    print(f'Steps for Power: {c}')
    ax1.plot(xs, pmfx_power[0, :N], '--', label=f'Power, N={N}')

    # solve via gauss-seidel method
    K_matrix, kappa = prepare_gs(A_matrix, b_vector, S_matrix, s_vector, N, xs)
    pmfx_gs, c = solve_gs(K_matrix, kappa, N, max_iter=1000)
    print(f'Steps for Gauss-Seidel: {c}')
    ax1.plot(xs, pmfx_gs[0, :N], '--', label=f'Gauss-Seidel, N={N}')

    # solve via jacobi method (first prepare, then solve).
    K_matrix, kappa = prepare_gs(A_matrix, b_vector, S_matrix, s_vector, N, xs)
    pmfx_jac, c = solve_power(K_matrix, kappa, N, max_iter=1000)
    print(f'Steps for Jacobi: {c}')
    ax1.plot(xs, pmfx_jac[0, :N], '--', label=f'Jacobi, N={N}')

    # solve via gaussian elimination
    I_AoG, h_v = prepare_ge(A_matrix, b_vector, S_matrix, s_vector, N, xs)
    pmfx_ge = solve_ge(I_AoG.copy(), h_v.copy(), N)
    ax1.plot(xs, pmfx_ge[0, :N], '--', label=f'Gaussian Elimination, N={N}')

    # solve via another form of ge, by using an additional step
    AA = create_combined_mat_for_gs(I_AoG.copy(), h_v.copy())
    pmfx_ge2 = solve_ge2(AA, N)
    ax1.plot(xs, pmfx_ge2[0, :N], '--', label=f'Gaussian Elimination 2.0, N={N}')

    # solve via an optimized form of ge
    pmfx_ge_opt = solve_ge_opt(I_AoG.copy(), h_v.copy(), N)
    ax1.plot(xs, pmfx_ge_opt[0, :N], '--', label=f'Gaussian Elimination Optimized, N={N}')

    # solve via another form of ge, not using I_AoG
    AoG, h_v = create_AoG_h(A_matrix, b_vector, S_matrix, s_vector, N, xs)
    pmfx_ge_AoG = solve_ge_AoG(AoG.copy(), h_v.copy(), N)
    ax1.plot(xs, pmfx_ge_AoG[0, :N], '--', label=f'Gaussian Elimination AoG, N={N}')

    # solve via an approximation method using ge with lu-decomposition (naive numpy solve)
    I_AoC = create_I_AoC_matrix(AoC, num_states)
    pmfx_lu_approx = solve_lu_decomp_approx(I_AoC, d_v)
    ax1.plot(xs, pmfx_lu_approx[0, :N], '--', label=f'LU Approx, N={N}')

    # solve via sampling
    NUM_SAMPLE_TRACES = 10000
    histograms = sample_smrm(num_states, xs, sampling_P_matrix, S_matrix, s_vector, N, max_iter=NUM_SAMPLE_TRACES)
    histograms[:] = histograms[:] / NUM_SAMPLE_TRACES   # normalization of the results is done externally.
    print("Sum of normalized histogram:", sum(histograms[:N]))
    ax1.plot(xs[:-1], histograms[:N - 1], '.', label=f'naive sampling with {NUM_SAMPLE_TRACES} samples')

    # solve via lu-decomp_approximation once more, but now using more points than previously
    # We find that changing N to a really large amount leads to a more precise solution. But
    # we also learn that the ge method lacks accuracy.
    N2 = 100000
    xs2 = np.arange(0, N2)
    AoC, d_v = prepare_power(A_matrix, b_vector, S_matrix, s_vector, N2, xs2)
    I_AoC = create_I_AoC_matrix(AoC, num_states)
    pmfx_LU_approx_larger = solve_lu_decomp_approx(I_AoC, d_v)
    ax1.plot(xs, pmfx_LU_approx_larger[0, :N], '--', label='LU Approx, N=' + str(N2))

    # plot results, resultant graph, and errors relative to the power method
    ax1.set_ylabel("pmf(x)")
    ax1.set_xlabel("Cumulated award")
    ax1.legend()

    bench = pmfx_power[0, :N]
    ax2.plot(xs, abs(bench - pmfx_jac[0, :N]), label=f'Jacobi,  N={N}')
    ax2.plot(xs, abs(bench - pmfx_gs[0, :N]), label=f'Gauss-Seidel,  N={N}')
    ax2.plot(xs, abs(bench - pmfx_ge_AoG[0, :N]), marker='x', label=f'GE AoG,  N={N}')
    ax2.plot(xs, abs(bench - pmfx_ge[0, :N]), label=f'GE,  N={N}')
    ax2.plot(xs, abs(bench - pmfx_lu_approx[0, :N]), label=f'LU approx, N={N}')
    ax2.plot(xs, abs(bench - pmfx_LU_approx_larger[0, :N]), label=f'LU approx  N={N2}')
    ax2.plot(xs, abs(bench - histograms[:N]), label=f'Sampling error, Num Samples={NUM_SAMPLE_TRACES}')

    ax2.legend()
    ax2.set_yscale('log')
    ax2.set_xlabel("Cumulated award")
    ax1.set_ylabel("log error (absolute diff. from power)")
    plt.show()
