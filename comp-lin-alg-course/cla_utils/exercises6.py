import numpy as np


def get_Lk(m, lvec):
    """Compute the lower triangular row operation mxm matrix L_k 
    which has ones on the diagonal, and below diagonal entries
    in column k given by lvec (k is inferred from the size of lvec).

    :param m: integer giving the dimensions of L.
    :param lvec: a m-k-1 dimensional numpy array.

    :return Lk: an mxm dimensional numpy array.

    """
    k = m - 1 - lvec.size
    Lk = np.eye(m)
    Lk[k+1:, k] = lvec
    return Lk


def LU_inplace(A, return_corners=False):
    """Compute the LU factorisation of A, using the in-place scheme so
    that the strictly lower triangular components of the array contain
    the strictly lower triangular components of L, and the upper
    triangular components of the array contain the upper triangular
    components of U.

    :param A: an mxm-dimensional numpy array

    """
    m, __ = A.shape
    if return_corners:
        uband = np.zeros(m)
        lband = np.zeros(m)
    for k in range(m-1):
        A[k+1:, k] /= A[k, k]   # compute column values of L below 1
        A[k+1:, k+1:] -= np.outer(A[k+1:, k], A[k, k+1:])  #
        if return_corners:
            uband[k] = A[0, m-1]
            lband[k] = A[m-1, 0]
    if return_corners:
        return uband, lband

def solve_L(L, b):
    """
    Solve systems Lx_i=b_i for x_i with L lower triangular, i=1,2,\ldots,k

    :param L: an mxm-dimensional numpy array, assumed lower triangular
    :param b: an mxk-dimensional numpy array, with ith column containing \
    b_i

    :return x: an mxk-dimensional numpy array, with ith column containing \
    the solution x_i

    """

    m, k = b.shape
    x = np.zeros((m, k), dtype=complex)
    x[0, :] = b[0, :]/L[0, 0]
    for i in range(1, m):
        x[i, :] = (b[i, :] - L[i, :i].dot(x[:i, :]))/L[i, i]
    return x


def solve_U(U, b, one_dim=False):
    """
    Solve systems Ux_i=b_i for x_i with U upper triangular, i=1,2,\ldots,k

    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing \
    b_i

    :return x: an mxk-dimensional numpy array, with ith column containing \
    the solution x_i

    """
    if one_dim:
        m = b.size
        x = np.zeros(m, dtype=complex)
        x[-1] = b[-1] / U[-1, -1]
        for i in range(m - 2, -1, -1):
            x[i] = (b[i] - U[i, i:].dot(x[i:])) / U[i, i]
        return x
    m, k = b.shape
    x = np.zeros((m, k), dtype=complex)
    x[-1, :] = b[-1, :]/U[-1, -1]
    for i in range(m-2, -1, -1):
        x[i, :] = (b[i, :] - U[i, i:].dot(x[i:, :]))/U[i, i]
    return x


def inverse_LU(A):
    """
    Form the inverse of A via LU factorisation.

    :param A: an mxm-dimensional numpy array.

    :return Ainv: an mxm-dimensional numpy array.

    """

    m, _ = A.shape
    LU_inplace(A)
    Uinv = solve_U(np.triu(A), np.eye(m, dtype=A.dtype))
    Linv = solve_L(A - np.triu(A) + np.eye(m), np.eye(m, dtype=A.dtype))
    Ainv = Uinv.dot(Linv)
    return Ainv


def solve_LU(A, b):
    """
    Solve Ax = b using LU decomposition
    @param A: mxm matrix A
    @param b: vector to solve for
    @return: m dimensional numpy array solution x
    """
    LU_inplace(A)
    m, _ = A.shape
    L = np.tril(A, -1)
    U = A - L
    L += np.eye(m)
    c = np.zeros((m, 1))
    Ux = solve_L(L, b)
    x = solve_U(U, Ux)
    return x.reshape((m,))

