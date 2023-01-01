import pytest
from q5 import *
import numpy as np


@pytest.mark.parametrize('M, N, deltat, deltax', [(50, 30, 0.1, 0.01), (20, 34, 0.02, 0.03)])
def test_compute_Uhat(M, N, deltat, deltax):
    """
    Test Uhat is correct
    @param M: M
    @param N: N
    @param deltat: deltat
    @param deltax: deltax
    """
    Rhat = np.random.rand(2*M*N) + 1.0j * np.random.rand(2*M*N)
    B, B21 = construct_B(deltat, deltax, M, return_B21=True)
    D1 = np.random.rand(N) + np.random.rand(N)*1.0j
    D2 = np.random.rand(N) + np.random.rand(N)*1.0j
    Uhat = compute_Uhat(M, N, Rhat, D1, D2, B21, deltat, deltax)
    I = np.eye(2*M)

    # check we have correctly eliminated pk and solved for U using pk and qk
    for k in range(N):
        assert np.linalg.norm((D1[k]*I + D2[k]*B).dot(Uhat[k*2*M:(k+1)*2*M]) - Rhat[k*2*M:(k+1)*2*M]) < 10e-6


@pytest.mark.parametrize('M, N, k, alpha', [(3, 3, 9, 0.01), (20, 34, 124, 0.003), (50, 66, 191, 0.07)])
def test_mult_VinvxI(M, N, k, alpha):
    """
    Check algorithm in part d works to compute (V^{-1} \otimes I)x
    @param M: M
    @param N: N
    @param k: k
    @param alpha: alpha
    """
    b = np.random.rand(2*M*N)
    V = construct_V(alpha, N)
    I = np.eye(2 * M)
    soln = left_mult_VinvxI(b, M, N, alpha)
    u = np.kron(np.linalg.inv(V), I).dot(b)
    # check this function computes the correct matrix product
    assert np.linalg.norm(soln - u) < 10e-6


@pytest.mark.parametrize('M, N, k, alpha', [(3, 3, 9, 1), (20, 34, 124, 0.003), (50, 66, 191, 0.07)])
def test_mult_VxI(M, N, k, alpha):
    """
    Check algorithm in part d works to compute (V \otimes I)x
    @param M: M
    @param N: N
    @param k: k
    @param alpha: alpha
    """
    b = np.random.rand(2*M*N)
    V = construct_V(alpha, N)
    I = np.eye(2 * M)
    soln = left_mult_VxI_matrix(b, M, N, alpha)
    u = np.kron(V, I).dot(b)
    # check this function computes the correct matrix product
    assert np.linalg.norm(soln - u) < 10e-6


@pytest.mark.parametrize('N, alpha', [(5, 0.01), (20, 0.003)])
def test_V(N, alpha):
    """
    Test V is correct
    @param N: N
    @param alpha: alpha
    """
    C1alpha = np.eye(N) - np.diag(np.ones(N - 1), -1)
    C1alpha[0, N - 1] = -alpha

    C2alpha = 0.5 * (np.eye(N) + np.diag(np.ones(N - 1), -1))
    C2alpha[0, N - 1] = 0.5 * alpha

    V = construct_V(alpha, N)

    # check vi eigenvectors of V
    for k in range(N):
        assert np.linalg.norm(C1alpha.dot(V[:, k]) - (1 - alpha**(1/N) * np.e**(2*(N-1)*np.pi*1.0j*k/N))) < 10e-6
        assert np.linalg.norm(C2alpha.dot(V[:, k]) - 0.5*(1 + alpha ** (1 / N) * np.e ** (2 * (N - 1) * np.pi * 1.0j * k / N))) < 10e-6


@pytest.mark.parametrize('N, alpha', [(5, 0.01), (20, 0.003)])
def test_V_and_Ds(N, alpha):
    """
    Test Vs and Ds are correct
    @param N: N
    @param alpha: alpha
    """
    C1alpha = np.eye(N) - np.diag(np.ones(N-1), -1)
    C1alpha[0, N-1] = -alpha

    C2alpha = 0.5*(np.eye(N) + np.diag(np.ones(N-1), -1))
    C2alpha[0, N-1] = 0.5*alpha

    V = construct_V(alpha, N)
    D1 = construct_D1(N, alpha)
    D2 = construct_D2(N, alpha)

    c1eigs = np.linalg.eig(C1alpha)[0]
    c2eigs = np.linalg.eig(C2alpha)[0]

    # check diagonal entries are eigenvalues
    assert np.max(np.abs(np.sort(c1eigs) - np.sort(D1))) < 10e-6
    assert np.max(np.abs(np.sort(c2eigs) - np.sort(D2))) < 10e-6

    # check diagonalisation
    assert np.linalg.norm(C1alpha - V.dot(D1).dot(np.linalg.inv(V))) < 10e-6
    assert np.linalg.norm(C2alpha - V.dot(D2).dot(np.linalg.inv(V))) < 10e-6


@pytest.mark.parametrize('M, N, deltat, deltax, alpha', [(100, 50, 0.01, 0.03, 0.001), (20, 34, 0.001, 0.02, 0.003), (50, 66, 0.1, 0.07, 0.1)])
def test_U(M, N, deltat, deltax, alpha):
    """
    Check U is the solution
    @param M: M
    @param N: N
    @param deltat: deltat
    @param deltax: deltax
    @param alpha: alpha
    """
    p0 = np.sin(np.linspace(0, np.pi, M))
    q0 = np.zeros(M)
    U0 = np.zeros(2*M*N)

    U = solve_system(p0, N, alpha)
    C1alpha = np.eye(N) - np.diag(np.ones(N-1), -1)
    C1alpha[0, N-1] = -alpha

    r = construct_r(p0, q0, deltat, deltat)
    B = construct_B(deltat, deltat, M)
    R = construct_R(M, N, r, alpha, B, U)

    C2alpha = 0.5*(np.eye(N) + np.diag(np.ones(N-1), -1))
    C2alpha[0, N-1] = 0.5*alpha
    I = np.eye(2*M)

    assert np.linalg.norm(np.dot(np.kron(C1alpha, I) + np.kron(C2alpha, B), U) - R) < 10e-3
