import numpy as np

from cla_utils import solve_L, solve_U


def perm(p, i, j):
    """
    For p representing a permutation P, i.e. Px[i] = x[p[i]],
    replace with p representing the permutation P_{i,j}P, where
    P_{i,j} exchanges rows i and j.

    :param p: an m-dimensional numpy array of integers.
    """
    p[[i, j]] = p[[j, i]]


def LUP_inplace(A, sign=False):
    """
    Compute the LUP factorisation of A with partial pivoting, using the
    in-place scheme so that the strictly lower triangular components
    of the array contain the strictly lower triangular components of
    L, and the upper triangular components of the array contain the
    upper triangular components of U.

    :param sign: a boolean value for whether or not to return the sign of the permutation p
    :param A: an mxm-dimensional numpy array

    :return p: an m-dimensional integer array describing the permutation \
    i.e. (Px)[i] = x[p[i]]
    :return sgn: the sign of the permutation p, either 1 or -1
    """

    m, _ = A.shape
    n = 0
    p = np.arange(m)
    sgn = 1
    for k in range(m - 1):
        i = np.argmax(np.abs(A[k:, k])) + k
        tmp = 1.0*A[k, :]
        A[k, :] = A[i, :]
        A[i, :] = tmp
        perm(p, i, k)
        if i != k:
            n += 1
        A[k + 1:, k] /= A[k, k]
        A[k + 1:, k + 1:] -= np.outer(A[k + 1:, k], A[k, k + 1:])  #
    sgn = (-1.0)**n
    if sign:
        return sgn
    return p


def solve_LUP(A, b):
    """
    Solve Ax=b using LUP factorisation.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an m-dimensional numpy array
    """
    m, _ = A.shape
    p = LUP_inplace(A)
    L = np.tril(A, -1)
    U = A - L
    L += np.eye(m)
    c = np.zeros((m, 1))
    c[:, 0] = b[p]
    Ux = solve_L(L, c)
    x = solve_U(U, Ux)
    return x.reshape((m,))


def det_LUP(A):
    """
    Find the determinant of A using LUP factorisation.

    :param A: an mxm-dimensional numpy array

    :return detA: floating point number, the determinant.
    """

    sgn = LUP_inplace(A, True)
    det = sgn * np.prod(A.diagonal())
    return det
