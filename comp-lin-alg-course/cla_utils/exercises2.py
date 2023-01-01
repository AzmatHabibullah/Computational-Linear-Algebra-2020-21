import numpy as np


def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """
    r = v
    m, n = Q.shape
    for i in range(n):
        r = r - Q[:, i].conjugate().dot(v) * Q[:, i]
    u = np.zeros(n, dtype=complex)
    for i in range(n):
        u[i] = Q[:, i].conjugate().dot(v)
    return r, u


def solveQ(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """
    m = b.size
    x = np.zeros(m, dtype=complex)
    for i in range(m):
        x = x + Q.conjugate().T[:, i] * b[i]
    return x


def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """

    return Q.dot(Q.conjugate().T)


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an lxm-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U.
    """
    m, n = V.shape
    fac = np.linalg.qr(V, 'complete')
    Q = fac[0]
    Qhat = Q[:, -(m-n):]
    return Qhat


def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """
    m, n = A.shape
    newA = np.copy(A)
    Q = np.zeros((m, n), A.dtype)
    R = np.zeros((n, n), A.dtype)
    for j in range(n):
        R[:j, j] = Q[:, :j].conjugate().T.dot(A[:, j])
        newA[:, j] = newA[:, j] - np.dot(Q[:, :j], R[:j, j])
        R[j, j] = np.linalg.norm(newA[:, j])
        Q[:, j] = newA[:, j]/R[j, j]   
    return Q, R


def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, producing
    :param A: mxn numpy array
    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    V = np.copy(A)
    Q = np.zeros((m, n), A.dtype)
    R = np.zeros((n, n), A.dtype)
    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        Q[:, i] = V[:, i]/R[i, i]
        R[i, i+1:n] = np.dot(Q[:, i].T.conj(), A[:, i+1:n])
        V[:, i+1:n] = V[:, i+1:n] - np.outer(Q[:, i], R[i, i+1:n])
    return Q, R

def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """

    raise NotImplementedError

    return R

def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    R = np.eye(n)
    for i in range(1,m):
        Rk = GS_modified_get_R(A, i)
        np.dot(A, Rk, out=A)
        np.dot(R, Rk, out=R)
    R = np.linalg.inv(R)
    return A, R
