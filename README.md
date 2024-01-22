## Background

This repository holds the code used to generate the results for the PhD thesis: [Algorithms for reachability problems on stochastic Markov reward models](https://etheses.bham.ac.uk/id/eprint/11842/)).
It provides the `pymrm` package that can be used to solve stochastic markov reward models, for discrete-lattice problems as well as continuous problems.

## Tutorial

Whilst the theory of SMRMs can be found in the linked thesis, a somewhat cryptic example of how to use the python package
can be found under the `/examples` folder.

Below is a shorter version of `waste_treatment_example.py`. It attempts to solve a simple discrete-lattice problem found in the paper: 
[Numerical approximation of probability mass functions via the inverse discrete fourier transform. Methodology and Computing in Applied Probability,
16(4):1025–1038, 2014](https://link.springer.com/article/10.1007/s11009-013-9366-3).

Some  imports and definitions:
```
import scipy.stats as st
import matplotlib.pyplot as plt
from pymrm.smrm import *

"""
We solve an semi-Markov Process (SMP) problem, represented as a discrete (lattice) sMRM.
"""

# variant of the discrete weibull distribution
def discrete_wei_variant_pmf(k, q, b):
    temp1 = q ** ((k - 1) ** b) - q ** (k ** b)
    temp2 = k - np.floor(k)
    temp3 = temp2 < 1e-15
    temp4 = k == 0
    temp1[temp4] = 0
    return temp1 * temp3
```

We proceed to recreate the SMP model from the paper using 3D hypermatrices:
```
if __name__ == "__main__":
    num_states, N = 2, 100
    A_matrix = np.asarray([[0, 1],
                           [0.95, 0]])
    b_vector = np.asarray([0, 0.05])

    pmf_matrix = np.zeros((num_states, num_states, N), )
    pmf_vector = np.zeros((num_states, N), )

    points = np.arange(0, N)
    pmf_matrix[0, 1] = st.geom.pmf(points, 0.8)
    pmf_matrix[1, 0] = discrete_wei_variant_pmf(points, 0.3, 0.5)
    pmf_vector[1] = discrete_wei_variant_pmf(points, 0.5, 0.7)
    xs = np.arange(0, N)
```
1. Firstly, we introduced the matrix `A_matrix` that represents state transitions `S_? x S_?` where `S_?` is the set of states not in the goal state set, but can reach the goal state. We also introduce a `b_vector`, where `b_vector[0]` is the probability of State `0` entering the goal state. These two matrices are in effect a representation of a Markov Chain.
2. Then, we introduce a `pmf_matrix` which is a matrix of probability mass functions. For example `pmf_matrix[0,1](x)` is the probability of transitioning between States `0 -> 1`, under time `x`.
3. The first two dimensions of `pmf_matrix` aligns with `A_matrix`, and similarly for `pmf_vector`. 
4. We define `xs` to be the range of values we want to solve for, i.e. `0,1,2,3,...,N`.

We now have enough parameters to solve the problem with Gaussian elimination using the `prepare_ge_pmfs` followed by `solve_ge` from the `pymrm` library:
```
    # solve via GE (prepare then solve)
    I_AoG, h_v = prepare_ge_pmfs(A_matrix, b_vector, pmf_matrix, pmf_vector, N)
    pmfx_ge = solve_ge(I_AoG.copy(), h_v.copy(), N)

    # plot results (pmf on one graph, and errors on the other).
    fig, axs = plt.subplots(1, 1)

    axs[0].plot(xs, pmfx_ge[0, :N], label='Gaussian elimination, N=' + str(N), linewidth=5, linestyle=':',
                color='black')
    axs[0].set_title("Cumulated time starting from state 0 reaching state 2")
    axs[0].set_ylabel("pmf(x)")
    axs[0].set_xlabel("time")
    axs[0].legend()
    plt.show()
```
A similar plot of the solution can be found in the Thesis, on page 74, under subsection 5.6.2 (Example 2: Waste treatment semi-Markov process). 


 