import pytest
from q1 import construct_tridiag, solve_tridiag_LU_inplace
from numpy import random
import numpy as np


@pytest.mark.parametrize('c, d, m', [(0, 3, 9), (3, 4, 5), (56, 72, 23), (-1.0j, 7 + 3.3j, 1923)])
def test_construct_tridiag(c, d, m):
    """
    Test tridiagonal construction
    @param c: diagonal entries
    @param d: super/subdiagonal entries
    @param m: matrix dimension
    """
    random.seed(1477*m)
    A = construct_tridiag(c, d, m)

    # check correct dimension
    assert(A.shape == (m, m))

    # check diagonal is c,  super and sub diagonals are d and other terms are 0
    A -= np.diag([c]*m, 0) + np.diag([d]*(m-1), 1) + np.diag([d]*(m-1), -1)
    assert(np.linalg.norm(A) < 10e-10)


@pytest.mark.parametrize('c, d, m, n', [(3, 6, 11, 4), (54, 22, 255, 18), (500 - 6.1j, np.pi + 3.0j, 1231, 633)])
def test_solve_tridiag_LU_inplace(c, d, m, n):
    """
    Test algorithm for question 1
    @param c: diagonal entries for A
    @param d: super/subdiagonal entries for A
    @param m: dimensions of A
    @param n: number of bi to solve for
    """
    random.seed(124*m)
    B = np.random.rand(m, n)
    A = construct_tridiag(c, d, m)
    X = solve_tridiag_LU_inplace(c, d, B, returnAll=False)

    # check each equation is solved
    assert(np.linalg.norm(A.dot(X) - B) < 10**-6)


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)