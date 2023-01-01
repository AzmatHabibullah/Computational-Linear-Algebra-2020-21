import cmath

import numpy as np
import scipy.linalg as sp

np.set_printoptions(linewidth=320)



def householder(A, kmax=None):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.

    :return R: an mxn-dimensional numpy array containing the upper \
    triangular matrix
    """
    m, n = A.shape
    if kmax is None:
        kmax = n

    R = np.copy(A)
    e1 = np.zeros(m)
    e1[0] = 1
    for k in range(kmax):
        x = R[k:, k]
        if np.iscomplex(x[0]) or x[0] == 0:
            sgn = np.e ** (1.0j * cmath.polar(x[0])[1])
        else:
            sgn = np.sign(x[0])
        v_k = sgn*np.linalg.norm(x)*e1[:m-k] + x
        v_k = v_k/np.linalg.norm(v_k)
        R[k:, k:] = R[k:, k:] - 2*np.outer(v_k, v_k.conjugate().T.dot(R[k:, k:]))
    return R


def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """
    Ab = np.concatenate((A, b), axis = 1) 
    m, n = A.shape
    RQstarB = householder(Ab, n)
    x = sp.solve_triangular(RQstarB[:, :n], RQstarB[:, m:])
    return x


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """
    m, n = A.shape
    AI = np.concatenate((A, np.eye(m)), axis = 1)
    RQstarI = householder(AI, n)
    R = RQstarI[:, :n]
    Q = RQstarI[:, -m:].conj().transpose()
    return Q, R


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """

    m, n = A.shape
    Ab = np.concatenate([A, np.array([b]).conj().T], axis = 1)
    RQstarb = householder(Ab, n)
    x = sp.solve_triangular(RQstarb[:n, :n], RQstarb[:n, -1])
    return x
