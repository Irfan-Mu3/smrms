from __future__ import division
import time
from sympy import *

"""
Experiment, from an earlier part of the thesis, involving symbolic strategies to solve SMRMs.
"""

if __name__ == '__main__':
    x, y, z, t = symbols('x y z t')
    M = Matrix([[x,2+x,3*x +5,x*2 + 10, 11*x-9, 8*x-20],
                [2+x,x,2+x,x, 27*x - 10, 15*x - 20],
                [x,2+x,x*5 - 3,x*30 + 1, x+50 -10, 11*x + 12*x**2],
                [x+5 - 10*x,20*x+ x**2 +x**3,x+8 -10,2+x, 42, 36*x],
                [x+5 - 10*x + 10*x**3, 6*x,x+8 -10 - 25*3 + x**2,10+x, -15*x, 16*x-4],
                [x + x**2 + 3,6*x + 12,x- 10,x*2,x**2,x-15]
                ])

    print(M)
    start = time.time()
    K = M.inv()
    print("Time for inversion:",time.time()-start)
    print(K)

    start = time.time()
    K = simplify(K)
    print("Time for simplification:",time.time()-start)

    print(K)