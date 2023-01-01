import pytest
from q1 import construct_tridiag
from q2 import solve_Ax_b, solve_Ax_b, solve_LU, construct_A
from cla_utils import *
import numpy as np


@pytest.mark.parametrize('c, d, m', [(0.001, -0.0113, 9), (3, 4., 5), (56 + 6j, 72.3 - 2.1j, 23)])
def test_solve_Ax_b(c, d, m):
    """
    Test the algorithm for solving Ax=b works
    @param c: diagonal entries
    @param d: super/subdiagonal entries
    @param m: matrix dimensions
    """
    random.seed(1477*m)
    A, T, u1, u2, v1, v2 = construct_A(m, c, d, return_all=True)

    b = np.random.rand(m)
    x = solve_Ax_b(c, d, b)

    # check correct dimension
    assert(x.size == m)

    # check close to numpy implementation
    assert(np.linalg.norm(np.linalg.solve(A, b) - x) < 10e-6)
