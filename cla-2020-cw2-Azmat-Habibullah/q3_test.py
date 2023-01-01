import pytest
from q3 import *
from cla_utils import *
from numpy import random
from q1 import *
import numpy as np


@pytest.mark.parametrize('m', [20, 40, 87])
def test_qr_factor_tri(m):
    """
    Test qr_factor_tri algorithm
    @param m: matrix dimensions
    """
    random.seed(1878*m)
    A = construct_tridiag(1.0*random.randint(m), random.randint(m), m)
    A0 = 1.0*A  # make a deep copy
    V = qr_factor_tri(A0, return_vecs=True)

    # check R is upper triangular
    assert(np.allclose(A0, np.triu(A0)))
    # check Q is orthogonal
    assert(np.linalg.norm(np.dot(A0.T, A0) - np.dot(A.T, A)) < 1.0e-6)


@pytest.mark.parametrize('m', [20, 40, 87])
def test_qr_alg_tri(m):
    """
    Test qr_alg_tri algorithm
    @param m: matrix dimensions
    """
    random.seed(1302*m)
    A = np.random.rand(m, m)
    A = A + A.T
    A0 = 1.0*A
    A2 = qr_alg_tri(A0)

    # check Hermitian
    assert(np.linalg.norm(A2 - np.conj(A2).T) < 1.0e-4)
    # check tridiagonal
    assert(np.linalg.norm(A2[np.tril_indices(m, -2)])/m**2 < 1.0e-5)
    assert(np.linalg.norm(A2[np.triu_indices(m, 2)])/m**2 < 1.0e-5)
    # check for conservation of trace
    assert(np.abs(np.trace(A0) - np.trace(A2)) < 1.0e-6)


@pytest.mark.parametrize('c, d, m', [(5., 3, 9), (3, 4., 5), (56, 72., 15)])
def test_concatenate(c, d, m):
    """
    Test concatenation for part e
    @param c: diagonal entries
    @param d: super/subdiagonal entries
    @param m: matrix dimensions
    """
    A = construct_tridiag(c, d, m)
    alg_iterates, pure_iterates = concatenate(A, compare_to_pure=True)

    # verify we drop below the tolerance the expected number of times
    assert np.sum(alg_iterates < 10e-12) >= m
    assert np.sum(pure_iterates < 10e-12) >= m


@pytest.mark.parametrize('m', [20, 40, 87])
def test_wilkinson(m):
    """
    Test Wilkinson shift
    @param m: matrix dimensions
    """
    random.seed(1302*m)
    A = np.random.rand(m, m)
    A = A + A.T
    A0 = 1.0*A
    A2 = qr_alg_tri(A0, shift=True)
    #check it is still Hermitian
    assert(np.linalg.norm(A2 - np.conj(A2).T) < 1.0e-4)
    #check for tridiagonal
    assert(np.linalg.norm(A2[np.tril_indices(m, -2)])/m**2 < 1.0e-5)
    assert(np.linalg.norm(A2[np.triu_indices(m, 2)])/m**2 < 1.0e-5)
    #check for conservation of trace
    assert(np.abs(np.trace(A0) - np.trace(A2)) < 1.0e-6)



