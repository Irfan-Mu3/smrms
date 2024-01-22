import numpy as np
import matplotlib.patches as mpatches
from pymrm.smrm_continuous import continuous_power_method_trapz_exact, continuous_prepare_AoC_no_pdfs, \
    continuous_prepare_AoC_no_pdf_w_s1

import matplotlib.pyplot as plt
from pacal import *
import scipy.integrate as sint
import pyfftw as fftw

fftw.interfaces.cache.enable()
pfftw = fftw.interfaces.numpy_fft.fft
ipfftw = fftw.interfaces.numpy_fft.ifft

"""This example solves the problem presented in the paper: An Introduction to Solving for Quantities of Interest in 
Finite-State Semi-Markov Processes, by Warr & Collins."""

# The problem is an SMP time-reachability problem.
def weibull_pdf(ks, gamma, thet):
    c = gamma / thet
    vals = c * ks ** (gamma - 1) * np.exp(-(ks ** (gamma) / thet))
    vals[abs(vals) == np.inf] = 0  # , not needed
    return vals

def create_weibull(gamma, thet):
    return lambda ks: weibull_pdf(ks, gamma, thet)

def create_delta(dx):
    return lambda ks: delta_dist_uni(ks, dx)

def delta_dist_uni(ks, dx):
    return (UniformDistr(0, dx) + UniformDistr(0, dx)).pdf(ks)


def zero_dist(ks):
    return np.zeros_like(ks)


if __name__ == "__main__":
    total_num_states, N = 9, 4001

    ######################################################
    # remember: With exception to the last three states, remaining in 1-5 is not possible, since they are not absorbing.
    a = 0
    b = 60 * 24

    xs = np.linspace(a, b, N)

    dx = (b - a) / (N - 1)
    print("dx:", dx)
    scaled_w = dx * np.asarray([1] * N)
    scaled_w[0] = 0

    f1 = create_weibull(4.738025, 4566277818.13)
    f2 = create_weibull(2.207438, 14541.6089)
    f3 = create_weibull(0.766338, 16.6991)
    f4 = create_weibull(2.303331, 1017649.5158)
    f6 = create_weibull(1.623492, 4707.3132)
    f_delta = create_delta(dx)

    P_matrix = np.asarray([[0.0000, 0.7447, 0.0084, 0.1339, 0.0042, 0.0063, 0.0000, 0.0063, 0.0962],
                           [0.0192, 0.0000, 0.0137, 0.0247, 0.0027, 0.0027, 0.0577, 0.8298, 0.0495],
                           [0.0000, 0.5833, 0.0000, 0.1667, 0.0833, 0.0000, 0.0000, 0.0000, 0.1667],
                           [0.0000, 0.0135, 0.0405, 0.0000, 0.0135, 0.0270, 0.0811, 0.7028, 0.1216],
                           [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
                           [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
                           [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
                           [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
                           [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
                           ])

    print("P_matrix", P_matrix)

    Rew_matrix = np.asarray([[f_delta, f1, f1, f1, f1, f2, f_delta, f2, f3],
                             [f4, f_delta, f1, f4, f1, f1, f4, f4, f6],
                             [f_delta, f4, f_delta, f1, f1, f_delta, f_delta, f_delta, f3],
                             [f_delta, f4, f4, f_delta, f4, f4, f4, f4, f6],
                             [f_delta] * 7 + [f4, f_delta],
                             [f_delta] * 7 + [f4, f_delta],
                             [f_delta] * 9,
                             [f_delta] * 9,
                             [f_delta] * 9,
                             ])

    max_val = max([max(Rew_matrix[i, j](xs)) for i in range(9) for j in range(9)])
    print("max value of S_matrix", max_val)
    assert np.isfinite(max_val), "max value is a singularity or nan"

    ######################################################

    all_survival_funcs = np.zeros((total_num_states + 1, N))
    survival_cumulated = np.ones_like(xs)
    absorbed_survival = np.ones_like(xs)
    all_cdf_funcs = np.zeros((total_num_states + 1, N))
    cdf_cumulated = np.zeros_like(xs)

    all_survival_funcs[-1] = np.ones(N)

    state_names = ['CCU', 'PCCU', 'ICU', 'MED', 'SURG', 'AMB', 'ECF', 'HOME', 'DIED']

    for i in [8, 7, 6, 5, 4, 3, 2, 1]:
        idx = i

        # reachable states: Start backwards and keep adding
        old_set = set()
        reachable_set = set(np.where(P_matrix[:, idx] != 0)[0])

        while sorted(old_set) != sorted(reachable_set):
            frontier = reachable_set.difference(old_set)
            old_set = reachable_set.copy()

            for i in frontier:
                states = np.where(P_matrix[:, i] != 0)[0]
                reachable_set.update(states)

        # reachable_set.update([6,7,8])
        reachable_set.discard(idx)
        list_reachable_set = list(reachable_set)
        list_unreachable_set = np.delete(np.arange(0, total_num_states), np.append(list_reachable_set, idx))

        # discard all states not reaching B as we know their reward contribution is zero, and it is not necessary
        # from a computational point of view.

        pre_A_matrix = P_matrix[list_reachable_set]
        A_matrix = pre_A_matrix[:, list_reachable_set]
        rem_P_matrix = pre_A_matrix[:, list_unreachable_set]

        s0_b_vector = np.sum(rem_P_matrix, 1)
        s1_b_vector = P_matrix[:, idx][list_reachable_set]

        print("A mat:", A_matrix)
        print("b_vec s0:", s0_b_vector)
        print("b_vec s1:", s1_b_vector)
        print("sum of remaining probabilities:", np.sum(A_matrix, 1) + s1_b_vector)

        assert np.allclose(
            np.sum(A_matrix, axis=1) + s0_b_vector + s1_b_vector, 1)

        S_matrix = Rew_matrix[list_reachable_set][:, list_reachable_set]
        s_vector = Rew_matrix[:, idx][list_reachable_set]

        #####################################################

        AoC_w, d_v = continuous_prepare_AoC_no_pdf_w_s1(A_matrix, s1_b_vector, S_matrix, s_vector, N, xs,
                                                        scaled_weights=scaled_w.copy())
        AoC, _ = continuous_prepare_AoC_no_pdf_w_s1(A_matrix, s1_b_vector, S_matrix, s_vector, N, xs,
                                                    scaled_weights=None)
        pdfx_trapz = continuous_power_method_trapz_exact(AoC, AoC_w, d_v, N, scaled_w.copy(), max_iter=2000,
                                                         trapz=True)
        sol = pdfx_trapz[0, :N]

        cdf = sint.cumtrapz(sol, xs, dx, initial=0)

        if idx in [8, 7, 6]:
            print("i", i)
            survival_cumulated -= cdf
            absorbed_survival -= cdf
            cdf_cumulated += cdf
        else:
            survival_cumulated -= (cdf * absorbed_survival)

        all_survival_funcs[idx] = survival_cumulated.copy()
        all_cdf_funcs[idx] = cdf_cumulated.copy()

    plt.ylim([0, 1])
    recs = []
    day_xs = xs / 24
    absorb_states = [6, 7, 8]
    state_colors_abs = ['black', 'grey', 'lightgrey']

    for i in range(len(absorb_states)):
        plt.fill_between(day_xs, all_cdf_funcs[absorb_states[i]], all_cdf_funcs[absorb_states[i] + 1],
                         color=state_colors_abs[i], )
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=state_colors_abs[i], edgecolor='black'))

    plt.ylabel("Probability")
    plt.xlabel("Length of stay (days) before discharge from hospital")
    plt.legend(recs, np.flip(state_names[6:]), loc='center right')
    plt.show()

    # TODO: Potentially wrong solutions. Need to be checked
    plt.xlim([0, b / 24])
    state_colors = ['firebrick', 'indianred', 'rosybrown', 'white', 'silver', 'darkgrey', 'grey', 'dimgrey', 'black']

    recs = []
    for i in range(0, len(state_colors)):
        plt.fill_between(day_xs, all_survival_funcs[i], all_survival_funcs[i + 1],
                         color=state_colors[i], )
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=state_colors[i], edgecolor='black'))

    plt.legend(recs, state_names, loc='center right')

    plt.title("")
    plt.ylabel("Probability")
    plt.xlabel("Time elapsed (in days)")
    plt.show()
