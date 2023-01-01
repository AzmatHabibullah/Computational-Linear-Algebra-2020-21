import pytest
from fit import *
from numpy import random
import numpy as np

from svd import make_gram, gram_svals, regularise


@pytest.mark.parametrize('m, n', [(20, 4), (40, 20), (70, 13)])
def test_arnoldi(m, n):
    """
    Test Arnoldi implementation is working
    :param m: matrix m dimension
    :param n: matrix n dimension
    """
    x = np.random.randn(m)
    X = np.diag(x)
    f = np.random.randn(m)

    Q, H, d = polyfitA(X, f, n)
    # check dimensions
    assert(Q.shape == (m, n+1))
    assert(H.shape == (n+1, n))
    # check norms are sqrt(m)
    assert(np.linalg.norm((Q.conj().T).dot(Q) - m*np.eye(n+1)) < 1.0e-6)
    # check 'almost' similarity
    assert(np.linalg.norm(X.dot(Q[:,:-1]) - Q.dot(H)) < 1.0e-6)


@pytest.mark.parametrize('m', [20, 40, 80])
def test_arnoldi_interpolation(m):
    """
    Test the Vandermonde + Arnoldi algorithm perfectly fits for the square case
    :param m: degree of polynomial and fit
    """
    x = np.cos(np.linspace(0, m, m) * np.pi / m)
    X = np.diag(x)
    f = np.random.randn(m)
    s = x

    Q, H, d = polyfitA(X, f, m-1)
    y = polyvalA(d, H, s)

    # check perfect interpolation
    assert(np.linalg.norm(f-y) < 1.0e-6)


@pytest.mark.parametrize('m', [5, 10, 20])
def test_vandermonde_interpolation(m):
    """
    Test the Vandermonde algorithm perfectly fits for the square case
    Lower values of m due to ill-conditioning effects
    :param m: degree of polynomial and fit
    """
    x = np.cos(np.linspace(0, m, m) * np.pi / m)
    f = np.random.randn(m)
    s = x

    c = polyfit(x, f, m-1)
    y = polyval(c, s)

    # check perfect interpolation
    assert(np.linalg.norm(f - y) < 1.0e-6)


@pytest.mark.parametrize('m', [10, 20, 30])
def test_gram(m):
    """
    Test Gram matrices are well constructed
    :param m: dimensions of matrix
    :return:
    """
    A = make_gram(m)
    s = gram_svals(m)
    # check Hermitian (symmetric as real in our case)
    assert np.linalg.norm(A.T - A) < 10e-6
    # check all singular values are eigenvalues
    assert np.linalg.norm(np.sort(np.linalg.eig(A)[0]) - np.sort(s)) < 10e-6
    # check all eigenvalues positive (semipositive definite)
    assert np.all(s>0)
    # check ill conditioned
    assert np.linalg.cond(A) > 10**(m/2 - 1)


@pytest.mark.parametrize('m, truncations', [(10,1), (20, 6), (80, 30)])
def test_regularise(m, truncations):
    """
    Check SVD truncation provides a good approximation
    :param m: dimensions of Gram matrix
    :param truncations: number of eigenvalues to remove
    :return:
    """
    A = make_gram(m)
    epsilon = gram_svals(m)[-truncations]
    A0 = regularise(A, epsilon)

    # check good approximation
    assert np.linalg.norm(A - A0) < 10e-6


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
