import matplotlib.pyplot as plt
import numpy as np


def example():
    m = 20
    A = np.zeros((m, m))
    np.fill_diagonal(A, np.linspace(1, 20, 20))
    coefficients = np.poly(A)
    for i in range(100):
        perturbations = np.random.normal(0, 1, m + 1)
        perturbed = np.poly(A)*(1 + 10**-10 * perturbations)
        plt.plot(np.roots(perturbed).real, np.roots(perturbed).imag)
    plt.show()

def Q1AQ1s(A):
    """
    For a matrix A, find the unitary matrix Q1 such that the first
    column of Q1*A has zeros below the diagonal. Then return A1 = Q1*A*Q1^*.

    :param A: an mxm numpy array

    :return A1: an mxm numpy array
    """

    R = np.copy(A)
    m, _ = A.shape
    e1 = np.zeros(m)
    e1[0] = 1
    x = R[:, 0]
    v_k = np.sign(np.sign(x[0]) + 0.5) * np.linalg.norm(x) * e1[:m] + x
    v_k = v_k / np.linalg.norm(v_k)
    R[:, :] -= 2 * np.outer(v_k, v_k.conjugate().T.dot(R[:, :]))
    R[:, :] -= 2 * np.outer(R[:, :].dot(v_k), v_k.conjugate().T)
    return R


def hessenberg(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.

    :param A: an mxm numpy array
    """

    m, _ = A.shape
    e1 = np.zeros(m)
    e1[0] = 1
    for k in range(m - 2):
        x = A[k+1:, k]
        v_k = np.sign(np.sign(x[0]) + 0.5) * np.linalg.norm(x) * e1[:m - (k+1)] + x
        norm = np.linalg.norm(v_k)
        if norm==0:
            norm = 1
        v_k = v_k / norm
        A[k+1:, k:] -= 2 * np.outer(v_k, v_k.conjugate().T.dot(A[k+1:, k:]))
        A[:, k+1:] -= 2 * np.outer(A[:, k+1:].dot(v_k), v_k.conjugate().T)


def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array
    
    :return Q: an mxm numpy array
    """

    m, _ = A.shape
    e1 = np.zeros(m)
    Q = np.eye(m)
    e1[0] = 1
    for k in range(m - 2):
        x = A[k + 1:, k]
        v_k = np.sign(np.sign(x[0]) + 0.5) * np.linalg.norm(x) * e1[:m - (k + 1)] + x
        v_k = v_k / np.linalg.norm(v_k)
        Q[k+1:, :] -= 2 * np.outer(v_k, v_k.conjugate().T.dot(Q[k + 1:, :])) # 0s
        A[k + 1:, k:] -= 2 * np.outer(v_k, v_k.conjugate().T.dot(A[k + 1:, k:]))
        A[:, k + 1:] -= 2 * np.outer(A[:, k + 1:].dot(v_k), v_k.conjugate().T)
    return Q.conjugate().T

def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvalues and eigenvectors.
    :param H: an mxm numpy array
    :return V: an mxm numpy array whose columns are the eigenvectors of H
    """
    m, n = H.shape
    assert(m==n)
    assert(np.linalg.norm(H[np.tril_indices(m, -2)]) < 1.0e-6)
    _, V = np.linalg.eig(H)
    return V


def ev(A):
    """
    Given a matrix A, return the eigenvectors of A. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).
    :param A: an mxm numpy array
    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """

    Q = hessenbergQ(A)
    h_ev = hessenberg_ev(A)
    return Q.dot(h_ev)