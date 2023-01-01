'''Tests for the eighth exercise set.'''
import pytest
import q4
from cla_utils import GMRES
from numpy import random
import numpy as np


@pytest.mark.parametrize('m', [20, 204, 18])
def test_upper_triangular_precond(m):
    """
    Test upper triangular precondition works
    @param m: matrix dimensions
    @return:
    """
    b = random.randn(m)
    _, U = np.linalg.qr(random.randn(m,m))
    apply_pc = q4.make_upper_triangular_pc(U)
    soln = apply_pc(b)
    err1 = b - np.dot(U, soln)
    # test we solve the system correctly
    assert(np.linalg.norm(err1) < 1.0e-6)
    A = random.randn(m, m)
    err2 = b - np.dot(A, soln)
    assert(np.linalg.norm(err2) > 1.0e-6)


@pytest.mark.parametrize('m', [20, 204, 18])
def test_preconditioned_GMRES(m):
    """
    Test preconditioned GMRES
    @param m: matrix dimensions
    """
    diagonal = np.random.choice(np.linspace(-100, 100, 500) + 1.0j*np.linspace(-100, 100, 500), m)
    A = np.random.rand(m, m) + 1.0j*np.random.rand(m, m)
    b = random.randn(m)
    _, U = np.linalg.qr(random.randn(m,m))
    apply_pc = q4.make_upper_triangular_pc(U)
    x, _ = GMRES(A, b, apply_pc=apply_pc, maxit=1000, tol=1.0e-3)
    # test GMRES solves the system Ax = b
    assert(np.linalg.norm(np.dot(A, x) - b) < 1.0e-3)


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
